"""Visualization callbacks for NPE training."""

from typing import Optional

import torch
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
