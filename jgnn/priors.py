"""Prior distributions for simulation-based inference."""
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.distributions as dist


class BoxUniform:
    """
    Box uniform prior distribution.

    A multivariate uniform distribution where each dimension has independent
    uniform bounds. Supports both NumPy and PyTorch sampling.

    Fixed parameters can be specified by setting min == max for that dimension.
    Fixed parameters will always return their fixed value during sampling.

    Parameters
    ----------
    prior_dict : dict
        Dictionary containing prior parameters. Should have keys:
        - 'labels' : list of str - Parameter names
        - 'min' : list or array - Minimum values for each parameter
        - 'max' : list or array - Maximum values for each parameter

        Alternatively, can specify bounds per parameter:
        - Each label as key with dict value containing 'min' and 'max'

        To fix a parameter, set min == max for that dimension.

    Examples
    --------
    >>> # Style 1: Global min/max arrays
    >>> prior_dict = {
    ...     'labels': ['param1', 'param2'],
    ...     'min': [0.0, -1.0],
    ...     'max': [1.0, 1.0]
    ... }
    >>> prior = BoxUniform(prior_dict)

    >>> # Style 2: Per-parameter specification
    >>> prior_dict = {
    ...     'param1': {'min': 0.0, 'max': 1.0},
    ...     'param2': {'min': -1.0, 'max': 1.0}
    ... }
    >>> prior = BoxUniform(prior_dict)

    >>> # Fixed parameters: set min == max
    >>> prior_dict = {
    ...     'labels': ['varying', 'fixed'],
    ...     'min': [0.0, 0.5],
    ...     'max': [1.0, 0.5]  # fixed at 0.5
    ... }
    >>> prior = BoxUniform(prior_dict)
    >>> samples = prior.sample(100)
    >>> # samples[:, 0] varies in [0, 1]
    >>> # samples[:, 1] is always 0.5
    """

    def __init__(self, prior_dict: Dict):
        self.prior_dict = prior_dict

        # Parse the prior_dict to extract labels, min, and max
        if 'labels' in prior_dict:
            # Style 1: Global specification
            self.labels = list(prior_dict['labels'])
            self.min = np.array(prior_dict['min'], dtype=np.float32)
            self.max = np.array(prior_dict['max'], dtype=np.float32)
        else:
            # Style 2: Per-parameter specification
            self.labels = []
            mins = []
            maxs = []
            for label, bounds in prior_dict.items():
                if isinstance(bounds, dict) and 'min' in bounds and 'max' in bounds:
                    self.labels.append(label)
                    mins.append(bounds['min'])
                    maxs.append(bounds['max'])
            self.min = np.array(mins, dtype=np.float32)
            self.max = np.array(maxs, dtype=np.float32)

        self.ndim = len(self.labels)

        # Validate
        if len(self.min) != self.ndim or len(self.max) != self.ndim:
            raise ValueError(
                f"Dimension mismatch: labels has {self.ndim} elements, "
                f"but min has {len(self.min)} and max has {len(self.max)}"
            )

        if np.any(self.min > self.max):
            raise ValueError("All min values must be less than or equal to max values")

        # Identify fixed parameters (where min == max)
        self.is_fixed = np.isclose(self.min, self.max, rtol=0, atol=1e-10)
        self.fixed_values = np.where(self.is_fixed, self.min, np.nan)

        # Count fixed vs varying parameters
        self.num_fixed = np.sum(self.is_fixed)
        self.num_varying = self.ndim - self.num_fixed

        # Print info about fixed parameters
        if self.num_fixed > 0:
            fixed_params = [self.labels[i] for i in range(self.ndim) if self.is_fixed[i]]
            print(f"[Prior] Fixed parameters: {fixed_params}")

        # Create torch distribution for log_prob computation
        # For fixed parameters, we use a small epsilon to avoid numerical issues
        # Use larger epsilon to avoid float32 precision issues
        epsilon = 1e-6
        max_adjusted = self.max.copy()
        for i in range(self.ndim):
            if self.is_fixed[i]:
                max_adjusted[i] = self.min[i] + epsilon

        min_torch = torch.tensor(self.min, dtype=torch.float32)
        max_torch = torch.tensor(max_adjusted, dtype=torch.float32)

        self._torch_dist = dist.Independent(
            dist.Uniform(min_torch, max_torch),
            reinterpreted_batch_ndims=1
        )

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from the box uniform prior.

        For fixed parameters (where min == max), the fixed value is returned.
        For varying parameters, values are sampled uniformly from [min, max].

        Parameters
        ----------
        num_samples : int
            Number of samples to draw
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        samples : np.ndarray, shape (num_samples, ndim)
            Samples from the prior
        """
        if seed is not None:
            np.random.seed(seed)

        samples = np.random.uniform(
            low=self.min,
            high=self.max,
            size=(num_samples, self.ndim)
        ).astype(np.float32)

        # Override fixed parameters with their fixed values
        if self.num_fixed > 0:
            for i in range(self.ndim):
                if self.is_fixed[i]:
                    samples[:, i] = self.fixed_values[i]

        return samples

    def log_prob(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute log probability of samples under the prior.

        For fixed parameters, checks if the value matches the fixed value
        (within tolerance). If not, returns -inf.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor, shape (..., ndim)
            Samples to evaluate

        Returns
        -------
        log_prob : np.ndarray or torch.Tensor, shape (...)
            Log probability of each sample. Returns -inf for samples outside bounds
            or not matching fixed parameter values.
        """
        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.tensor(x, dtype=torch.float32)
            return_numpy = True
        else:
            x_torch = x
            return_numpy = False

        # Compute log probability from uniform distributions
        log_p = self._torch_dist.log_prob(x_torch)

        # Check fixed parameters
        if self.num_fixed > 0:
            fixed_values_torch = torch.tensor(self.fixed_values, dtype=torch.float32, device=x_torch.device)
            is_fixed_torch = torch.tensor(self.is_fixed, dtype=torch.bool, device=x_torch.device)

            # For each fixed dimension, check if x matches the fixed value
            # Use a small tolerance for numerical comparison
            tolerance = 1e-6
            for i in range(self.ndim):
                if self.is_fixed[i]:
                    # Check if x[:, i] is close to fixed_values[i]
                    matches = torch.abs(x_torch[..., i] - fixed_values_torch[i]) < tolerance
                    # Set log_p to -inf where fixed parameter doesn't match
                    log_p = torch.where(matches, log_p, torch.tensor(float('-inf'), device=x_torch.device))

        # Convert back to numpy if needed
        if return_numpy:
            return log_p.numpy()
        return log_p

    def is_within_bounds(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Check if samples are within the prior bounds.

        For fixed parameters, checks if the value matches the fixed value
        (within tolerance).

        Parameters
        ----------
        x : np.ndarray or torch.Tensor, shape (..., ndim)
            Samples to check

        Returns
        -------
        mask : np.ndarray or torch.Tensor, shape (...)
            Boolean mask indicating if each sample is within bounds
        """
        if isinstance(x, np.ndarray):
            # Check normal bounds
            within_min = np.all(x >= self.min, axis=-1)
            within_max = np.all(x <= self.max, axis=-1)
            within_bounds = within_min & within_max

            # Check fixed parameters
            if self.num_fixed > 0:
                tolerance = 1e-6
                for i in range(self.ndim):
                    if self.is_fixed[i]:
                        # Check if x[:, i] matches the fixed value
                        matches = np.abs(x[..., i] - self.fixed_values[i]) < tolerance
                        within_bounds = within_bounds & matches

            return within_bounds
        else:
            min_torch = torch.tensor(self.min, dtype=x.dtype, device=x.device)
            max_torch = torch.tensor(self.max, dtype=x.dtype, device=x.device)
            within_min = torch.all(x >= min_torch, dim=-1)
            within_max = torch.all(x <= max_torch, dim=-1)
            within_bounds = within_min & within_max

            # Check fixed parameters
            if self.num_fixed > 0:
                fixed_values_torch = torch.tensor(self.fixed_values, dtype=x.dtype, device=x.device)
                tolerance = 1e-6
                for i in range(self.ndim):
                    if self.is_fixed[i]:
                        # Check if x[:, i] matches the fixed value
                        matches = torch.abs(x[..., i] - fixed_values_torch[i]) < tolerance
                        within_bounds = within_bounds & matches

            return within_bounds

    def __repr__(self):
        fixed_info = f", fixed={self.num_fixed}" if self.num_fixed > 0 else ""
        return (
            f"BoxUniform(ndim={self.ndim}, "
            f"varying={self.num_varying}"
            f"{fixed_info}, "
            f"labels={self.labels})"
        )

    def to_dict(self) -> Dict:
        """
        Convert prior to dictionary format.

        Returns
        -------
        prior_dict : dict
            Dictionary representation of the prior
        """
        return {
            'labels': self.labels,
            'min': self.min.tolist(),
            'max': self.max.tolist()
        }
