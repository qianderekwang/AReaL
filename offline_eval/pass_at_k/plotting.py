from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .aggregator import VersionPassSummary


def plot_version_summaries(
    summaries: list[VersionPassSummary],
    output_path: str | Path,
) -> None:
    if not summaries:
        raise ValueError("No version summaries available for plotting")

    output_path = Path(output_path)
    versions = [summary.tail_version for summary in summaries]
    metric_keys = sorted(
        {
            key
            for summary in summaries
            for key in summary.pass_metrics.keys()
        }
    )
    if not metric_keys:
        raise ValueError("No pass@k metrics available for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric_key in metric_keys:
        ys = [summary.pass_metrics.get(metric_key) for summary in summaries]
        ax.plot(versions, ys, marker="o", label=metric_key)

    ax.set_xlabel("tail_version")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k by version")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
