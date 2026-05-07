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

    # Metric definitions: (label, value, unit, full_scale_value, color, formatter)
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

    # --- Helper: single-card horizontal bar ---
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

    # --- Helper: big-number card ---
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

    # Row 0
    # (0, 0) Parse failure rate
    ax0 = fig.add_subplot(gs[0, 0])
    _horizontal_bar(ax0, "Parse Failure Rate", parse_failure, 1.0, "%", color="#d62728")

    # (0, 1) Inference error count
    ax1 = fig.add_subplot(gs[0, 1])
    ecolor = "#2ca02c" if error_count == 0 else "#d62728"
    _big_number(ax1, "Inference Errors", error_count, color=ecolor)

    # (0, 2) Samples evaluated / max
    ax2 = fig.add_subplot(gs[0, 2])
    _horizontal_bar(
        ax2, "Samples Evaluated", total_evaluated, max_samples, "samples", color="#1f77b4",
    )

    # Row 1
    # (1, 0) Average inference time
    ax3 = fig.add_subplot(gs[1, 0])
    # use a sensible full scale — 2× the value, but at least 10 s
    scale_inf = max(avg_inference * 2, 10)
    _horizontal_bar(ax3, "Avg Inference Time", avg_inference, scale_inf, "s", color="#ff7f0e")

    # (1, 1) Total runtime (in minutes)
    ax4 = fig.add_subplot(gs[1, 1])
    total_min = total_runtime / 60.0
    scale_run = max(total_min * 1.3, 5)
    _horizontal_bar(ax4, "Total Runtime", total_min, scale_run, "min", color="#9467bd")

    # (1, 2) Overall accuracy (cross-reference)
    ax5 = fig.add_subplot(gs[1, 2])
    _horizontal_bar(ax5, "Overall Accuracy", overall_acc, 1.0, "%", color="#17becf")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz]  Runtime metrics chart saved to {output_path}")


def plot_fallback_analysis(
    trajectories_path: str = "results/trajectories.jsonl",
    output_path: str = "results/fallback_analysis.png",
) -> None:
    """Read ``trajectories.jsonl`` and render the extraction cascade breakdown.

    Produces a two-panel figure:
      - **Top**: a single horizontal stacked bar showing what fraction of
        samples passed through each stage of the extraction pipeline:
        JSON-first-try, retry-with-2×-tokens, regex-fallback, and
        unrecoverable failure.
      - **Bottom**: per-subject grouped bars showing the regex-fallback
        rate and retry rate for each MMMU subject.
    """
    if not os.path.exists(trajectories_path):
        print(
            f"[viz]  No trajectories found at {trajectories_path} — "
            "skipping fallback chart."
        )
        return

    # Load and classify every record.
    records: list[dict] = []
    with open(trajectories_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("[viz]  No valid records in trajectories — skipping fallback chart.")
        return

    total = len(records)

    # Classify each sample into one of four mutually exclusive buckets.
    json_first = 0       # succeeded on first try, JSON only, no retry, no regex
    retry_ok = 0         # succeeded after retry (JSON), no regex needed
    regex_ok = 0         # succeeded only via regex fallback
    total_fail = 0       # all attempts failed

    per_subject: dict[str, dict[str, int]] = {}  # subject → {json, retry, regex, fail}

    for r in records:
        subj = r.get("subject", "unknown")
        if subj not in per_subject:
            per_subject[subj] = {"json": 0, "retry": 0, "regex": 0, "fail": 0, "n": 0}

        per_subject[subj]["n"] += 1

        if not r.get("extraction_succeeded", False):
            total_fail += 1
            per_subject[subj]["fail"] += 1
        elif r.get("used_fallback_extraction", False):
            regex_ok += 1
            per_subject[subj]["regex"] += 1
        elif r.get("retry_used", False):
            retry_ok += 1
            per_subject[subj]["retry"] += 1
        else:
            json_first += 1
            per_subject[subj]["json"] += 1

    # Build the figure layout.
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 2.5], hspace=0.45)
    fig.suptitle(
        "Qwen3-VL-2B — Extraction Cascade Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: stacked horizontal bar (overall cascade breakdown).
    ax_top = fig.add_subplot(gs[0])

    stages = [
        (json_first, "JSON 1ˢᵗ try", "#2ca02c"),
        (retry_ok, "Retry 2× tkns", "#ff7f0e"),
        (regex_ok, "Regex fallback", "#d62728"),
        (total_fail, "Unrecovered ❌", "#7f7f7f"),
    ]

    left = 0.0
    bar_height = 0.55
    for count, label, color in stages:
        frac = count / total if total > 0 else 0.0
        ax_top.barh(0, frac, bar_height, left=left, color=color,
                    edgecolor="white", linewidth=1.2)
        if frac > 0.03:
            ax_top.text(
                left + frac / 2, 0,
                f"{label}\n{count} ({frac * 100:.1f}%)",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if color not in ("#2ca02c", "#ff7f0e") else "white",
            )
        left += frac

    ax_top.set_xlim(0, 1.02)
    ax_top.set_ylim(-0.6, 0.6)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_title(
        f"Extraction cascade (n={total})", fontsize=11, fontweight="bold", pad=10
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    # Panel 2: per-subject grouped bars (retry & regex rates).
    ax_bot = fig.add_subplot(gs[1])

    subjects = sorted(per_subject.keys())
    n_subjects = len(subjects)

    x = range(n_subjects)
    width = 0.35

    retry_rates = [
        (per_subject[s]["retry"] / per_subject[s]["n"] * 100)
        if per_subject[s]["n"] > 0 else 0.0
        for s in subjects
    ]
    regex_rates = [
        (per_subject[s]["regex"] / per_subject[s]["n"] * 100)
        if per_subject[s]["n"] > 0 else 0.0
        for s in subjects
    ]

    bars1 = ax_bot.bar(
        [xi - width / 2 for xi in x], retry_rates, width,
        color="#ff7f0e", edgecolor="white", linewidth=0.8, label="Retry rate"
    )
    bars2 = ax_bot.bar(
        [xi + width / 2 for xi in x], regex_rates, width,
        color="#d62728", edgecolor="white", linewidth=0.8, label="Regex fallback rate"
    )

    # Annotate bars
    for bar, val in zip(bars1, retry_rates):
        if val > 0:
            ax_bot.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
    for bar, val in zip(bars2, regex_rates):
        if val > 0:
            ax_bot.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(subjects, rotation=30, ha="right", fontsize=10)
    ax_bot.set_ylabel("Rate (%)", fontsize=11)
    ax_bot.set_title("Per-subject retry & fallback rates", fontsize=11, fontweight="bold", pad=8)
    ax_bot.legend(fontsize=9, loc="upper right")
    ax_bot.set_ylim(0, max(max(retry_rates), max(regex_rates)) * 1.5 + 6)

    # Save the figure.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz]  Fallback analysis chart saved to {output_path}")
