import os
import numpy as np
import matplotlib.pyplot as plt


def plot_fold_metrics(
    all_fold_metrics: list[dict],
    title: str = "SGC Baseline: 3-Fold Cross-Validation Metrics",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot per-fold and average metrics bar charts.

    Parameters
    ----------
    all_fold_metrics:
        List of metric dicts, one per fold (output of ``compute_metrics``).
    title:
        Super-title for the figure.
    save_path:
        If provided, the figure is saved to this path before being shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = [
        "#ff6b6b", "#6ab04c", "#686de0", "#f6c667",
        "#63cdda", "#4cd137", "#e84393", "#fd9644",
    ]

    metric_keys = list(all_fold_metrics[0].keys())
    metric_labels = [
        k.replace("(", " (") if "(" in k and k[3] != " " else k
        for k in metric_keys
    ]

    mean_values = [np.mean([f[k] for f in all_fold_metrics]) for k in metric_keys]
    std_values  = [np.std ([f[k] for f in all_fold_metrics]) for k in metric_keys]

    fold_titles = [f"Fold {i + 1}" for i in range(len(all_fold_metrics))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_order = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # Per-fold subplots
    for ax, fold, fold_title in zip(ax_order[:len(all_fold_metrics)], all_fold_metrics, fold_titles):
        values = [fold[k] for k in metric_keys]
        bars = ax.bar(metric_labels, values, color=colors[:len(metric_keys)])
        ax.set_title(fold_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Metric", fontsize=10)
        ax.set_ylabel("Value",  fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7,
            )

    # Average subplot with error bars
    ax_avg = ax_order[3]
    bars = ax_avg.bar(
        metric_labels, mean_values,
        yerr=std_values, capsize=5,
        color=colors[:len(metric_keys)],
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )
    ax_avg.set_title("Avg. Across Folds", fontsize=13, fontweight="bold")
    ax_avg.set_xlabel("Metric", fontsize=10)
    ax_avg.set_ylabel("Value",  fontsize=10)
    ax_avg.tick_params(axis="x", rotation=35)
    for bar, mean, std in zip(bars, mean_values, std_values):
        ax_avg.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + bar.get_height() * 0.01,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=7,
        )

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
