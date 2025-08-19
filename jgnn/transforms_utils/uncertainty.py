
import torch
from scipy.stats import truncnorm

class UncertaintySampler:
    """
    Samples uncertainty parameters from specified distributions and applies
    Gaussian noise to data along a specified axis.
    """
    def __init__(self, distribution_type, feature_idx, **params):
        """
        Args:
            distribution_type: 'uniform' or 'gaussian'
            **params: Parameters for the distribution
                For 'uniform': low, high (range for uniform sampling of std)
                For 'gaussian': mean, std (parameters for Gaussian sampling of std)
        """
        self.distribution_type = distribution_type
        self.feature_idx = feature_idx
        self.params = params

        # Validate parameters
        if distribution_type == 'uniform':
            if 'low' not in params or 'high' not in params:
                raise ValueError("Uniform distribution requires 'low' and 'high' parameters")
            if params['low'] < 0 or params['high'] < 0:
                raise ValueError("Uncertainty parameters must be non-negative")
            if params['low'] > params['high']:
                raise ValueError("'low' must be <= 'high' for uniform distribution")

        elif distribution_type == 'gaussian':
            if 'mean' not in params or 'std' not in params:
                raise ValueError("Gaussian distribution requires 'mean' and 'std' parameters")
            if params['mean'] < 0 or params['std'] < 0:
                raise ValueError("Uncertainty parameters must be non-negative")

        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

    def sample_uncertainty_std(self, n_samples=1):
        """
        Sample uncertainty standard deviation values.

        Args:
            n_samples: Number of std values to sample

        Returns:
            std_values: Tensor of sampled standard deviation values (no gradient)
        """
        if self.distribution_type == 'uniform':
            low = self.params['low']
            high = self.params['high']
            std_values = torch.rand(n_samples) * (high - low) + low

        elif self.distribution_type == 'gaussian':
            mean = self.params['mean']
            std = self.params['std']

            # Use truncated Gaussian with lower bound at 0
            # Define truncation bounds in terms of standard deviations from mean
            a = (0 - mean) / std  # Lower bound in standardized form
            b = float('inf')      # Upper bound (no upper limit)

            # Sample from truncated normal
            samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples)
            std_values = torch.tensor(samples, dtype=torch.float32)

        return std_values.detach()

    def __call__(self, batch):
        """
        Apply Gaussian uncertainty to a batch at a specific feature index and append
        the uncertainty values as a new feature.

        Args:
            batch: Batch object containing node features in batch.x

        Returns:
            batch: Modified batch with uncertainty applied to specified feature index
                batch.x will have an additional column for uncertainty std values.
        """
        batch = batch.clone()  # Ensure we don't modify the original batch
        data = batch.x

        if data.dim() != 2:
            raise ValueError(f"Expected 2D tensor [N_batch, N_dim], got {data.dim()}D")

        if self.feature_idx >= data.shape[1] or self.feature_idx < 0:
            raise ValueError(
                f"feature_idx {self.feature_idx} out of range for tensor with {data.shape[1]} features")

        # Sample uncertainty and apply to the specified feature index
        std_values = self.sample_uncertainty_std(len(data))  # sample
        noise = torch.normal(0, std_values)
        data[:, self.feature_idx] += noise # Apply noise to the specified feature

        # Append uncertainty values as a new feature column
        # make sure that the true noise is not included in the original data
        std_column = std_values.unsqueeze(1)  # Shape: [N_batch, 1]
        data = torch.cat([data, std_column], dim=1)

        batch.x = data
        return batch

    def get_distribution_info(self):
        """Return information about the current distribution configuration."""
        return {
            'distribution_type': self.distribution_type,
            'parameters': self.params
        }

