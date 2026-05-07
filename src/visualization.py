"""Visualization of evaluation results."""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")  # non-interactive backend — no display required


def plot_accuracy_chart(
    summary_path: str = "results/summary.json",
    output_path: str = "results/accuracy_chart.png",
) -> None:
    """Read summary.json and generate a bar chart of accuracy metrics.

    The first bar displays overall accuracy (highlighted in red), followed by
    one bar per subject from the per-subject breakdown. All values are shown
    as percentages with annotations on each bar.
    """
    if not os.path.exists(summary_path):
        print(f"[viz]  No summary found at {summary_path} — skipping chart.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    overall = summary.get("overall_accuracy", 0.0)
    per_subject = summary.get("per_subject_accuracy", {})

    if not per_subject:
        print("[viz]  No per-subject data found — skipping chart.")
        return

    # Build ordered lists: "Overall" first, then subjects in sorted order
    labels = ["Overall"] + sorted(per_subject.keys())
    values = [overall * 100] + [per_subject[s] * 100 for s in sorted(per_subject.keys())]

    # Colors: distinct color for "Overall", uniform color for subjects
    colors = ["#d62728"] + ["#1f77b4"] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(len(labels) * 0.7 + 2, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate each bar with the percentage value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Qwen3-VL-2B — MMMU Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2 + 2)

    # Rotate x-axis labels and adjust layout so they aren't truncated
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz]  Accuracy chart saved to {output_path}")


def plot_runtime_metrics(
    summary_path: str = "results/summary.json",
    output_path: str = "results/runtime_metrics.png",
) -> None:
    """Read ``summary.json`` and generate a dashboard of runtime & quality metrics.

    Renders a 2×3 grid showing:
      - parse failure rate (horizontal bar)
      - inference error count (big number)
      - completion ratio (samples evaluated / max)
      - average inference time (horizontal bar)
      - total runtime (horizontal bar, in min)
      - overall accuracy (horizontal bar, for cross-reference)
    """
    if not os.path.exists(summary_path):
        print(f"[viz]  No summary found at {summary_path} — skipping runtime chart.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # ------------------------------------------------------------------
    # Metric definitions  (label, value, unit, full_scale_value, color, formatter)
    # ------------------------------------------------------------------
    parse_failure = summary.get("parse_failure_rate", 0.0)
    error_count = summary.get("inference_error_count", 0)
    total_evaluated = summary.get("total_samples_evaluated", 0)
    max_samples = summary.get("max_samples", 0)
    avg_inference = summary.get("average_inference_time_sec", 0.0)
    total_runtime = summary.get("total_runtime_sec", 0.0)
    overall_acc = summary.get("overall_accuracy", 0.0)

    # Build a compact figure with GridSpec
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)
    fig.suptitle(
        "Qwen3-VL-2B — Runtime & Quality Metrics",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ---- Helper to draw a single-card horizontal bar ----
    def _horizontal_bar(
        ax,
        title: str,
        value: float,
        full: float,
        unit: str,
        color: str = "#2ca02c",
    ) -> None:
        """Draw a filled horizontal bar relative to *full*, with the value printed."""
        frac = min(value / full, 1.0) if full > 0 else 0.0
        ax.barh(0, frac, height=0.6, color=color, edgecolor="white", linewidth=0.8)
        ax.barh(0, 1.0, height=0.6, color="#e0e0e0", zorder=0)  # background
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        # Format the displayed number
        if unit == "%":
            label = f"{value * 100:.1f}%"
        elif unit == "s":
            label = f"{value:.2f} s"
        elif unit == "min":
            label = f"{value:.2f} min"
        else:
            label = f"{value:.1f} {unit}"
        ax.text(
            frac / 2, 0, label,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white" if frac > 0.3 else "#333",
        )
        # Show the scale value on the right
        full_label = f"{full * 100:.0f}%" if unit == "%" else f"{full:.1f} {unit}"
        ax.text(1.01, 0, full_label, va="center", fontsize=8, color="#666")

    # ---- Helper to draw a big-number card ----
    def _big_number(ax, title: str, value, color: str = "#1f77b4") -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.text(
            0.5, 0.45, str(value),
            ha="center", va="center",
            fontsize=36, fontweight="bold", color=color,
        )

    # ------------------------------------------------------------------
    # Row 0
    # ------------------------------------------------------------------
    # (0,0) Parse failure rate
    ax0 = fig.add_subplot(gs[0, 0])
    _horizontal_bar(ax0, "Parse Failure Rate", parse_failure, 1.0, "%", color="#d62728")

    # (0,1) Inference error count
    ax1 = fig.add_subplot(gs[0, 1])
    ecolor = "#2ca02c" if error_count == 0 else "#d62728"
    _big_number(ax1, "Inference Errors", error_count, color=ecolor)

    # (0,2) Samples evaluated / max
    ax2 = fig.add_subplot(gs[0, 2])
    _horizontal_bar(
        ax2, "Samples Evaluated", total_evaluated, max_samples, "samples", color="#1f77b4",
    )

    # ------------------------------------------------------------------
    # Row 1
    # ------------------------------------------------------------------
    # (1,0) Average inference time
    ax3 = fig.add_subplot(gs[1, 0])
    # use a sensible full scale — 2× the value, but at least 10 s
    scale_inf = max(avg_inference * 2, 10)
    _horizontal_bar(ax3, "Avg Inference Time", avg_inference, scale_inf, "s", color="#ff7f0e")

    # (1,1) Total runtime — show in minutes
    ax4 = fig.add_subplot(gs[1, 1])
    total_min = total_runtime / 60.0
    scale_run = max(total_min * 1.3, 5)
    _horizontal_bar(ax4, "Total Runtime", total_min, scale_run, "min", color="#9467bd")

    # (1,2) Overall accuracy (cross-reference)
    ax5 = fig.add_subplot(gs[1, 2])
    _horizontal_bar(ax5, "Overall Accuracy", overall_acc, 1.0, "%", color="#17becf")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz]  Runtime metrics chart saved to {output_path}")
