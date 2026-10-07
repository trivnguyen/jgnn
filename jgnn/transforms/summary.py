"""Radius-binned velocity moments as graph-level summary features.

`BinnedMoments` reads the node features `x = [log10 R, v_los, sigma_v]`
that `GetNodeFeatures` + `UncertaintySampler` leave on the batch, splits
each graph's stars into `n_bins` equal-count radial bins, and writes one
row per graph to `batch.summary`: for every bin the median log radius,
the log rms velocity, the log rms measurement error, the raw kurtosis
of the velocities and the log star count. The error correction is left
to the network (the rms error is given next to the rms velocity), so
no column ever saturates on a noise-dominated bin. Fixed physical
scalings bring every column to order unity, so no `norm_dict` entry
is needed.

It runs after the selection function and the uncertainty sampler, so
the summaries describe the same noisy, subselected stars the GNN sees;
on real data, where `x` is given, it works the same way.
"""

import math

import torch

# Reason: fixed scalings instead of a norm_dict entry. Dispersions span
# ~1-50 km/s, kurtosis ~1.5-6, and 2-50 stars per bin; these map each
# column to roughly [-2, 2] for the whole prior.
LOG_SIGMA_LOC, LOG_SIGMA_SCALE = 0.8, 0.6
KURT_LOC, KURT_SCALE = 3.0, 1.0
LOG_N_LOC, LOG_N_SCALE = 1.0, 0.5


class BinnedMoments:
    """Append radius-binned dispersion and kurtosis summaries to a batch.

    Args:
        n_bins: Equal-count radial bins per graph.
        r_idx, v_idx, err_idx: Columns of `batch.x` holding log10 R, the
            line-of-sight velocity and its measurement sigma.
        eps_kms: Floor on the dispersion [km/s] before taking logs.
    """

    N_PER_BIN = 5

    def __init__(self, n_bins: int = 4, r_idx: int = 0, v_idx: int = 1,
                 err_idx: int = 2, eps_kms: float = 0.01):
        self.n_bins = n_bins
        self.r_idx = r_idx
        self.v_idx = v_idx
        self.err_idx = err_idx
        self.eps_kms = eps_kms

    @property
    def output_size(self) -> int:
        """Length of one graph's summary row."""
        return self.N_PER_BIN * self.n_bins

    def _one(self, x: torch.Tensor) -> torch.Tensor:
        """Summary row for the stars `x` of one graph."""
        logr = x[:, self.r_idx]
        v = x[:, self.v_idx]
        err = x[:, self.err_idx] if x.shape[1] > self.err_idx else (
            torch.zeros_like(v))
        order = torch.argsort(logr)
        chunks = torch.tensor_split(order, self.n_bins)
        v = v - v.mean()  # Reason: Jeans fits marginalise a systemic v.
        eps2 = self.eps_kms ** 2
        rows = []
        for idx in chunks:
            n = idx.numel()
            if n == 0:
                rows.append(torch.zeros(self.N_PER_BIN, device=x.device))
                continue
            vb, eb = v[idx], err[idx]
            m2 = (vb ** 2).mean().clamp_min(eps2)
            e2 = (eb ** 2).mean().clamp_min(eps2)
            kurt = ((vb ** 4).mean() / m2 ** 2).clamp(1.0, 10.0)
            log_n = torch.tensor(math.log10(float(n)), device=x.device)
            rows.append(torch.stack([
                logr[idx].median(),
                (0.5 * torch.log10(m2) - LOG_SIGMA_LOC) / LOG_SIGMA_SCALE,
                (0.5 * torch.log10(e2) - LOG_SIGMA_LOC) / LOG_SIGMA_SCALE,
                (kurt - KURT_LOC) / KURT_SCALE,
                (log_n - LOG_N_LOC) / LOG_N_SCALE,
            ]))
        return torch.cat(rows)

    def __call__(self, batch):
        batch = batch.clone()
        if hasattr(batch, 'ptr') and batch.ptr is not None:
            bounds = batch.ptr.tolist()
        else:
            bounds = [0, batch.x.shape[0]]
        rows = [self._one(batch.x[a:b]) for a, b in zip(bounds[:-1],
                                                         bounds[1:])]
        batch.summary = torch.stack(rows)
        return batch


class OracleGaussianLogLike:
    """Per-knot profile log-likelihood of a Gaussian LOSVD as a summary.

    For a Gaussian (vdisp) twin the per-star likelihood is
    N(v_i | 0, s(R_i) + e_i^2) with s = sigma_los^2. Put knots R_k on a
    log grid and spread every star over its two neighbouring knots with
    linear weights w_ik in log R. For each knot and each value s_j of a
    log grid in sigma^2,

        f_kj = -1/2 sum_i w_ik [ v_i^2 / (s_j + e_i^2) + ln(s_j + e_i^2) ].

    Then sum_k f_k(s(R_k; theta)) is the galaxy log-likelihood up to a
    theta-independent constant and an O(h^2) quadrature error, so the
    table is a sufficient statistic for the posterior: the network only
    has to learn the Jeans map theta -> sigma_los^2(R_k) and read the
    table. Every knot row is shifted to its own maximum (the constants
    cancel), clipped at `-clip` and divided by `scale`; knots without
    stars are all zero. Velocities are used as they are: the twins have
    no systemic velocity, and neither does their exact likelihood.

    Args:
        n_knots: Knots in log10 R [kpc], evenly spaced over `log_r_range`
            (the 500k twin set spans 3e-4 to 20 kpc; stars outside the
            range go to the edge knot).
        log_r_range: (min, max) of log10 R [kpc] for the knots.
        n_sigma: Values of sigma^2, log-spaced over `log_s2_range`
            [km^2/s^2] (twin dispersions and errors span 0.01-40 km/s).
            This grid limits the accuracy: read back by linear
            interpolation in log sigma^2, 40 values miss the exact
            log-likelihood by up to 0.3 nats across a posterior, 160
            by 0.04 (npe_gap/check_oracle_summary.py).
        log_s2_range: (min, max) of log10 sigma^2.
        clip: Floor on a knot row relative to its own maximum [nats].
        scale: Divisor applied after clipping.
        r_idx, v_idx, err_idx: Columns of `batch.x` holding log10 R, the
            line-of-sight velocity and its measurement sigma.
    """

    def __init__(self, n_knots: int = 64, log_r_range=(-3.0, 1.2),
                 n_sigma: int = 160, log_s2_range=(-4.0, 3.4),
                 clip: float = 30.0, scale: float = 10.0, r_idx: int = 0,
                 v_idx: int = 1, err_idx: int = 2):
        self.n_knots = n_knots
        self.n_sigma = n_sigma
        self.log_r0, self.log_r1 = float(log_r_range[0]), float(
            log_r_range[1])
        self.log_knots = torch.linspace(self.log_r0, self.log_r1, n_knots)
        self.s2 = 10 ** torch.linspace(float(log_s2_range[0]),
                                       float(log_s2_range[1]), n_sigma)
        self.clip = clip
        self.scale = scale
        self.r_idx, self.v_idx, self.err_idx = r_idx, v_idx, err_idx

    @property
    def output_size(self) -> int:
        """Length of one graph's summary row."""
        return self.n_knots * self.n_sigma

    def table(self, x: torch.Tensor, graph: torch.Tensor,
              n_graphs: int) -> torch.Tensor:
        """Unscaled f_kj [n_graphs, n_knots, n_sigma] for stars `x`."""
        logr = x[:, self.r_idx]
        v = x[:, self.v_idx]
        err = x[:, self.err_idx] if x.shape[1] > self.err_idx else (
            torch.zeros_like(v))
        step = (self.log_r1 - self.log_r0) / (self.n_knots - 1)
        pos = ((logr - self.log_r0) / step).clamp(0.0,
                                                  self.n_knots - 1 - 1e-6)
        k0 = pos.floor().long()
        w1 = pos - k0.to(pos.dtype)
        w0 = 1.0 - w1
        k1 = (k0 + 1).clamp(max=self.n_knots - 1)
        tot = self.s2.to(x.device)[None, :] + err[:, None] ** 2
        g = -0.5 * (v[:, None] ** 2 / tot + torch.log(tot))
        f = torch.zeros(n_graphs * self.n_knots, self.n_sigma,
                        device=x.device, dtype=g.dtype)
        base = graph * self.n_knots
        f.index_add_(0, base + k0, w0[:, None] * g)
        f.index_add_(0, base + k1, w1[:, None] * g)
        return f.view(n_graphs, self.n_knots, self.n_sigma)

    def __call__(self, batch):
        batch = batch.clone()
        graph = batch.get('batch')
        if graph is None:
            graph = torch.zeros(batch.x.shape[0], dtype=torch.long,
                                device=batch.x.device)
        n_graphs = int(graph.max().item()) + 1 if graph.numel() else 1
        f = self.table(batch.x, graph, n_graphs)
        f = f - f.max(dim=2, keepdim=True).values
        batch.summary = (f.clamp(min=-self.clip) / self.scale).reshape(
            n_graphs, -1)
        return batch
