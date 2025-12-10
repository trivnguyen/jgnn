"""Plotting utilities for JGNN."""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wandb


def plot_corner(
    samples: np.ndarray,
    labels: List[str],
    title: Optional[str] = None,
    truths: Optional[np.ndarray] = None,
    quantiles: List[float] = [0.16, 0.5, 0.84],
    show_titles: bool = True,
    title_fmt: str = '.4f',
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """Create a corner plot for posterior samples.

    Args:
        samples: Posterior samples of shape (n_samples, n_params)
        labels: Parameter labels
        title: Overall title for the figure
        truths: True parameter values (optional)
        quantiles: Quantiles to show in 1D histograms
        show_titles: Whether to show parameter statistics in titles
        title_fmt: Format string for title values
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object
    """
    try:
        import corner
    except ImportError:
        raise ImportError(
            "corner package is required for corner plots. "
            "Install it with: pip install corner"
        )

    # Set default figure size if not provided
    n_params = samples.shape[1]
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    # Create corner plot
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=quantiles,
        show_titles=show_titles,
        title_fmt=title_fmt,
        title_kwargs={"fontsize": 12},
        truths=truths,
        fig=plt.figure(figsize=figsize)
    )

    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=14)

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Plotting] Corner plot saved to: {save_path}")

    return fig


def plot_corner_with_wandb(
    samples: np.ndarray,
    labels: List[str],
    round_num: int,
    wandb_logger,
    truths: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> plt.Figure:
    """Create corner plot and log to wandb.

    Args:
        samples: Posterior samples of shape (n_samples, n_params)
        labels: Parameter labels
        round_num: Current round number
        wandb_logger: WandB logger instance
        truths: True parameter values (optional)
        save_dir: Directory to save the plot (optional)

    Returns:
        matplotlib Figure object
    """
    # Determine save path
    save_path = None
    if save_dir is not None:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"corner_round{round_num}.png")

    # Create corner plot
    title = f"Posterior Samples - Round {round_num}"
    fig = plot_corner(
        samples=samples,
        labels=labels,
        title=title,
        truths=truths,
        save_path=save_path
    )

    # Log to wandb
    if wandb_logger is not None and wandb_logger.experiment is not None:
        wandb_logger.log_image(
            key=f"corner_plot/round_{round_num}",
            images=[fig]
        )
        print(f"[Plotting] Corner plot logged to WandB for round {round_num}")

    return fig


def plot_posterior_statistics(
    samples: np.ndarray,
    labels: List[str],
    title: Optional[str] = None,
    truths: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """Print and optionally save posterior statistics.

    Args:
        samples: Posterior samples of shape (n_samples, n_params)
        labels: Parameter labels
        title: Title for the statistics output
        truths: True parameter values (optional)
        save_path: Path to save statistics as text file (optional)
    """
    if title:
        print(f"\n{title}")
        print("=" * 80)

    stats_text = []
    stats_text.append(f"{'Parameter':<25} {'Median':>10} {'16th %':>10} {'84th %':>10} {'Mean':>10} {'Std':>10}")
    stats_text.append("-" * 80)

    for i, label in enumerate(labels):
        mean = samples[:, i].mean()
        std = samples[:, i].std()
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])

        line = f"{label:<25} {q50:10.4f} {q16:10.4f} {q84:10.4f} {mean:10.4f} {std:10.4f}"

        if truths is not None:
            truth = truths[i]
            line += f" (true={truth:10.4f})"

        stats_text.append(line)
        print(line)

    print("=" * 80 + "\n")

    # Save to file if requested
    if save_path is not None:
        with open(save_path, 'w') as f:
            if title:
                f.write(f"{title}\n")
                f.write("=" * 80 + "\n")
            f.write("\n".join(stats_text))
            f.write("\n")
        print(f"[Plotting] Statistics saved to: {save_path}")


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')
