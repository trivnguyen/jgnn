"""Visualization callbacks for NPE training."""

from typing import Optional, Sequence

import astropy.units as u
import torch
from torch_geometric.data import Batch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tarp


class NPEVisualizationCallback(pl.Callback):
    """Callback for visualizing NPE results during training.

    This callback generates diagnostic plots at regular intervals during training
    to monitor the quality of posterior estimation. Supported visualizations include:
    - Median posterior vs. true parameters (calibration plot)
    - TARP (Test of Accuracy with Random Points) coverage diagnostic
    - Rank statistics histogram

    Args:
        plot_every_n_epochs: Generate plots every N epochs
        n_posterior_samples: Number of posterior samples to draw per validation example
        n_val_samples: Maximum number of validation samples to use for plotting
        plot_median_v_true: Whether to plot median posterior vs. true parameters
        plot_tarp: Whether to plot TARP coverage diagnostic
        plot_rank: Whether to plot rank statistics histogram
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        n_posterior_samples: int = 500,
        n_val_samples: int = 1000,
        plot_median_v_true: bool = True,
        plot_tarp: bool = True,
        plot_rank: bool = True,
        use_default_mplstyle: bool = True,
    ):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.n_posterior_samples = n_posterior_samples
        self.n_val_samples = n_val_samples
        self.plot_median_v_true = plot_median_v_true
        self.plot_tarp = plot_tarp
        self.plot_rank = plot_rank

        if use_default_mplstyle:
            self._set_mplstyle()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate and log visualization plots at end of validation epoch.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The LightningModule being trained (NPE model)
        """
        # Check if we should plot this epoch
        if (trainer.current_epoch + 1) % self.plot_every_n_epochs != 0:
            return
        if not (self.plot_median_v_true or self.plot_tarp or self.plot_rank):
            return

        # Get the validation dataloader and extract a subset for visualization
        loader = trainer.val_dataloaders  # only support single val dataloader

        # Collect validation samples
        all_batches = []
        total_samples = 0
        for batch in loader:
            all_batches.append(batch)
            total_samples += batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
            if total_samples >= self.n_val_samples:
                break

        # Concatenate batches
        if len(all_batches) == 0:
            return

        # Extract true parameters
        all_true = torch.cat([b.theta for b in all_batches], dim=0)[:self.n_val_samples]

        # Sample from posterior
        pl_module.eval()
        with torch.no_grad():
            all_samples_list = []
            for batch in all_batches:
                batch_samples = pl_module.sample_from_batch(
                    batch, self.n_posterior_samples
                )
                all_samples_list.append(batch_samples.cpu())
            all_samples = torch.cat(all_samples_list, dim=0)[:self.n_val_samples]

        # Move to CPU for plotting
        all_true = all_true.cpu()

        # Create figures
        log_data = {'epoch': trainer.current_epoch}

        if self.plot_median_v_true:
            fig_median = self._plot_median_posterior(all_true, all_samples)
            log_data["figures/median_v_true"] = wandb.Image(fig_median)
            plt.close(fig_median)

        if self.plot_tarp:
            fig_tarp = self._plot_tarp(all_true, all_samples)
            log_data["figures/tarp"] = wandb.Image(fig_tarp)
            plt.close(fig_tarp)

        if self.plot_rank:
            fig_rank = self._plot_rank(all_true, all_samples)
            log_data["figures/rank"] = wandb.Image(fig_rank)
            plt.close(fig_rank)

        # Log to wandb
        if trainer.logger is not None:
            trainer.logger.experiment.log(log_data)

        plt.close('all')

    def _plot_median_posterior(self, true_params, samples):
        """Plot median posterior estimates vs. true parameter values.

        Creates scatter plots comparing the median of the posterior samples
        to the true parameter values, with a 1:1 reference line.

        Args:
            true_params: True parameter values, shape (n_samples, n_params)
            samples: Posterior samples, shape (n_samples, n_posterior_samples, n_params)

        Returns:
            matplotlib.figure.Figure: Figure containing the plots
        """
        median_posterior = torch.median(samples, dim=1)[0]  # (n_samples, n_params)
        p68_lower = torch.quantile(samples, 0.16, dim=1)
        p68_upper = torch.quantile(samples, 0.84, dim=1)
        n_params = true_params.shape[1]

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # Scatter plot with error bars
            x = true_params[:, i]
            y = median_posterior[:, i]
            yerr = [(median_posterior[:, i] - p68_lower[:, i]),
                    (p68_upper[:, i] - median_posterior[:, i])]
            ax.errorbar(
                x, y, yerr=yerr, fmt='o', alpha=0.3, markersize=2,
                capsize=1)

            # 1:1 line
            min_val = min(true_params[:, i].min(), median_posterior[:, i].min())
            max_val = max(true_params[:, i].max(), median_posterior[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, lw=2)

            # calculate R2 coefficient
            ss_res = torch.sum((y - x) ** 2)
            ss_tot = torch.sum((x - torch.mean(x)) ** 2)
            r2 = 1 - ss_res / ss_tot
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=12)

            ax.set_xlabel(f'True Parameter {i}')
            ax.set_ylabel(f'Median Posterior {i}')
            ax.set_title(f'Parameter {i}')
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        return fig

    def _plot_tarp(self, true_params, samples):
        """Plot TARP (Test of Accuracy with Random Points) coverage diagnostic.

        TARP tests whether the posterior credible regions have the correct coverage
        by comparing expected vs. observed coverage probabilities.

        Args:
            true_params: True parameter values, shape (n_samples, n_params)
            samples: Posterior samples, shape (n_samples, n_posterior_samples, n_params)

        Returns:
            matplotlib.figure.Figure: Figure containing the TARP plot
        """
        # Convert to numpy for tarp package
        true_params_np = true_params.numpy()
        samples_np = samples.numpy().transpose(1, 0, 2)

        # Compute TARP with bootstrapping
        ecp_bootstrap, alpha = tarp.get_tarp_coverage(
            samples_np, true_params_np, norm=True, metric="euclidean",
            references="random", bootstrap=True
        )
        ecp_mean = ecp_bootstrap.mean(0)
        ecp_std = ecp_bootstrap.std(0)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Plot expected coverage (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=2, label='Ideal')

        # Plot observed coverage
        ax.plot(alpha, ecp_mean, color='C0', lw=2, label='Observed')
        for k in [1, 2, 3]:
            ax.fill_between(
                alpha, ecp_mean - k * ecp_std, ecp_mean + k * ecp_std,
                color='C0', alpha=0.2
            )

        ax.set_xlabel('Credibility Level')
        ax.set_ylabel('Expected Coverage')
        ax.set_title('TARP Coverage')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        return fig

    def _plot_rank(self, true_params, samples):
        """Plot rank statistics histogram for posterior calibration.

        For well-calibrated posteriors, the rank of the true parameter within
        the posterior samples should be uniformly distributed.

        Args:
            true_params: True parameter values, shape (n_samples, n_params)
            samples: Posterior samples, shape (n_samples, n_posterior_samples, n_params)

        Returns:
            matplotlib.figure.Figure: Figure containing the rank histograms
        """
        n_params = true_params.shape[1]
        n_posterior_samples = samples.shape[1]

        # Compute ranks for each parameter
        ranks = torch.zeros(true_params.shape[0], n_params)
        for i in range(true_params.shape[0]):
            for j in range(n_params):
                # Count how many posterior samples are less than true value
                ranks[i, j] = (samples[i, :, j] < true_params[i, j]).sum()

        # Create figure
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # Plot histogram
            ax.hist(ranks[:, i].numpy(), bins=20, density=True, alpha=0.7, edgecolor='black')

            # Expected uniform distribution
            expected_height = 1.0 / n_posterior_samples
            ax.axhline(expected_height, color='k', linestyle='--', alpha=0.3, lw=2,
                      label='Ideal (uniform)')

            ax.set_xlabel(f'Rank')
            ax.set_ylabel('Density')
            ax.set_title(f'Parameter {i}')
            ax.legend()

        plt.tight_layout()
        return fig

    def _set_mplstyle(self):
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['axes.titlepad'] = 10
        mpl.rcParams['figure.facecolor'] = 'w'
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['xtick.minor.size'] = 5
        mpl.rcParams['xtick.minor.visible'] = True
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['ytick.minor.size'] = 5
        mpl.rcParams['ytick.minor.visible'] = True
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['axes.grid.axis'] = 'both'
        mpl.rcParams['axes.grid.which'] = 'major'
        mpl.rcParams['grid.linestyle'] = '--'
        mpl.rcParams['grid.alpha'] = 0.2
        mpl.rcParams['grid.color'] = 'black'
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['legend.fontsize'] = 12


class TargetPosteriorCallback(pl.Callback):
    """Plot a corner posterior on a real observed dataset each validation epoch.

    Loads the target galaxy catalog once at fit-start using the same
    kinematic_io pipeline as sample_preprocess_target.py, builds a single
    PyG Data graph (same icrs.create_graph_from_icrs path), then samples the
    NPE posterior and logs a corner plot to WandB at regular intervals.

    Args:
        catalog_path   : Path to the observed kinematic catalog.
        meta_key       : Galaxy key for kinematic_io.load_meta (e.g. 'draco_1').
        source         : Catalog source string for load_kinematic_data.
        loader_kwargs  : Extra kwargs forwarded to load_kinematic_data
                         (e.g. mem_prob_min, vlos_abs_max).
        cond_values    : Dict mapping each cond label name to its scalar value
                         for this galaxy (e.g. {'stellar_log_r_star': -0.64}).
                         Must match the cond_labels order used during training.
        cond_labels    : Ordered sequence of cond label names, same as
                         config.cond_labels used in training.
        n_posterior_samples : Number of posterior samples to draw each epoch.
        plot_every_n_epochs : Frequency of plotting.
        param_names    : Human-readable axis labels for the corner plot, one
                         per label.  Defaults to the label index numbers.
        meta_path      : Optional path to a custom meta CSV for load_meta.
        use_default_mplstyle : Apply the shared rcParams style.
    """

    def __init__(
        self,
        catalog_path: str,
        meta_key: str,
        source: str,
        loader_kwargs: Optional[dict] = None,
        cond_values: Optional[dict] = None,
        cond_labels: Optional[Sequence[str]] = None,
        n_posterior_samples: int = 2000,
        plot_every_n_epochs: int = 1,
        param_names: Optional[Sequence[str]] = None,
        meta_path: Optional[str] = None,
        use_default_mplstyle: bool = True,
    ):
        super().__init__()
        self.catalog_path = catalog_path
        self.meta_key = meta_key
        self.source = source
        self.loader_kwargs = loader_kwargs or {}
        self.cond_values = cond_values or {}
        self.cond_labels = list(cond_labels) if cond_labels else []
        self.n_posterior_samples = n_posterior_samples
        self.plot_every_n_epochs = plot_every_n_epochs
        self.param_names = param_names
        self.meta_path = meta_path
        self._graph = None

        if use_default_mplstyle:
            self._set_mplstyle()

    # ------------------------------------------------------------------
    # Fit start: load observed data and build the PyG graph once
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        from dsph_analysis import kinematic_io
        from jgnn.datasets.icrs import create_graph_from_icrs

        # Load metadata and catalog (same calls as sample_preprocess_target.py)
        if self.meta_path:
            meta = kinematic_io.load_meta(self.meta_key, meta_path=self.meta_path)
        else:
            meta = kinematic_io.load_meta(self.meta_key)

        data = kinematic_io.load_kinematic_data(
            self.catalog_path, meta=meta, source=self.source,
            **self.loader_kwargs)

        # Extract observables
        ra = data.ra.to_value(u.deg)
        dec = data.dec.to_value(u.deg)
        vlos = data.vlos.to_value(u.km / u.s)
        R_proj = data.R_proj.to_value(u.kpc)
        vlos_err = data.vlos_err.to_value(u.km / u.s)

        # Ordered cond vector (same ordering as cond_labels in training).
        # Any label absent from self.cond_values is derived from meta.
        if self.cond_labels:
            cond_values = dict(self.cond_values)
            for label in self.cond_labels:
                if label not in cond_values:
                    cond_values[label] = self._cond_from_meta(meta, label)
            cond = [cond_values[k] for k in self.cond_labels]
        else:
            cond = None

        # Build graph — no theta (real data has no ground-truth labels)
        graph = create_graph_from_icrs(
            ra, dec, vlos, R_proj, vlos_err=vlos_err,
            label=None, cond=cond)

        # Apply normalization from the trained model's norm_dict
        norm_dict = pl_module.norm_dict
        if norm_dict is not None:
            # x normalisation (currently identity in icrs.py but respected here)
            if 'x_loc' in norm_dict:
                x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32)
                x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32)
                graph.x = (graph.x - x_loc) / x_scale

            # cond normalisation
            if self.cond_labels and graph.cond is not None and 'cond_loc' in norm_dict:
                cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32)
                cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32)
                graph.cond = (graph.cond - cond_loc) / cond_scale

        self._graph = graph
        print(f"[TargetPosteriorCallback] Loaded {len(ra)} stars "
              f"from '{self.meta_key}' ({self.catalog_path})")

    # ------------------------------------------------------------------
    # Validation epoch end: sample posterior and log corner plot
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_every_n_epochs != 0:
            return
        if self._graph is None:
            return

        device = next(pl_module.parameters()).device
        batch = Batch.from_data_list([self._graph]).to(device)

        pl_module.eval()
        with torch.no_grad():
            # Returns shape (1, n_samples, n_params) in normalised theta space
            samples_norm = pl_module.sample_from_batch(
                batch, self.n_posterior_samples)

        # Squeeze the single-galaxy batch dimension → (n_samples, n_params)
        if samples_norm.dim() == 3:
            samples_norm = samples_norm.squeeze(0)

        # Denormalise to physical parameter space
        norm_dict = pl_module.norm_dict
        theta_loc = torch.tensor(norm_dict['theta_loc'],   dtype=torch.float32)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32)
        samples_phys = (samples_norm.cpu() * theta_scale + theta_loc).numpy()

        fig = self._plot_corner(samples_phys)
        if trainer.logger is not None:
            trainer.logger.experiment.log({
                'figures/target_posterior': wandb.Image(fig),
                'epoch': trainer.current_epoch,
            })
        plt.close(fig)

    # ------------------------------------------------------------------
    # Meta → cond value resolver
    # ------------------------------------------------------------------

    @staticmethod
    def _cond_from_meta(meta, label: str) -> float:
        """Derive a scalar cond value from DwarfMeta for a known label name."""
        if label == 'stellar_r_star':
            return meta.rhalf_kpc.value
        elif label == 'stellar_log_r_star':
            return float(np.log10(meta.rhalf_kpc.value))
        elif label == 'log_mwolf':
            return float(meta.log_mass_wolf)
        else:
            raise ValueError(
                f"Cannot auto-derive cond label '{label}' from meta. "
                f"Provide it explicitly via cond_values.")

    # ------------------------------------------------------------------
    # Corner plot
    # ------------------------------------------------------------------

    def _plot_corner(self, samples: np.ndarray):
        """Make a corner plot of posterior samples in physical space.

        Uses the ``corner`` package when available; falls back to a plain
        matplotlib triangle plot otherwise.

        Parameters
        ----------
        samples : ndarray, shape (n_posterior_samples, n_params)
        """
        n_params = samples.shape[1]
        labels = (list(self.param_names)
                  if self.param_names is not None
                  else [f'param_{i}' for i in range(n_params)])

        try:
            import corner
            fig = corner.corner(
                samples,
                labels=labels,
                show_titles=True,
                title_kwargs={'fontsize': 12},
                quantiles=[0.16, 0.5, 0.84],
                title_fmt='.3f',
            )
        except ImportError:
            fig, axes = plt.subplots(
                n_params, n_params, figsize=(3 * n_params, 3 * n_params))
            for i in range(n_params):
                for j in range(n_params):
                    ax = axes[i, j]
                    if j > i:
                        ax.set_visible(False)
                    elif i == j:
                        ax.hist(samples[:, i], bins=40, density=True, color='C0')
                        ax.set_xlabel(labels[i])
                        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
                        ax.set_title(
                            f'{labels[i]}\n'
                            f'${q50:.3f}_{{-{q50-q16:.3f}}}^{{+{q84-q50:.3f}}}$',
                            fontsize=10)
                    else:
                        ax.scatter(samples[:, j], samples[:, i],
                                   s=1, alpha=0.2, color='C0', rasterized=True)
                        ax.set_xlabel(labels[j])
                        ax.set_ylabel(labels[i])
            plt.tight_layout()

        return fig

    def _set_mplstyle(self):
        mpl.rcParams['font.size'] = 14
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['figure.facecolor'] = 'w'
