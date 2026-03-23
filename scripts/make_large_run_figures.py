#!/usr/bin/env python3
"""
Generate progress and example figures for the large-split Burgers surrogate run.

Usage:
    python3 scripts/make_large_run_figures.py \
        --workspace /tmp/autoresearch-codex-large.woja6k \
        --output-dir artifacts/2026-03-23-large-run
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ExperimentRow:
    index: int
    commit: str
    val_rel_l2: float
    memory_gb: float
    status: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/autoresearch-codex-large.woja6k"),
        help="Path to the isolated large-run workspace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/2026-03-23-large-run"),
        help="Directory to write figures and the summary JSON.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[ExperimentRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for index, row in enumerate(reader, start=1):
            rows.append(
                ExperimentRow(
                    index=index,
                    commit=row["commit"],
                    val_rel_l2=float(row["val_rel_l2"]),
                    memory_gb=float(row["memory_gb"]),
                    status=row["status"],
                    description=row["description"],
                )
            )
    if not rows:
        raise RuntimeError(f"No experiment rows found in {path}")
    return rows


def run_id_for_index(index: int) -> str:
    if index == 1:
        return "burgers-surrogate-large-baseline"
    return f"burgers-surrogate-large-exp{index:02d}"


def resolve_run_bundle(workspace: Path, index: int) -> tuple[Path, Path, dict]:
    fetched_root = workspace / "results" / "fetched_runs"
    matches = sorted(fetched_root.glob(f"{run_id_for_index(index)}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one fetched run for experiment {index}, found {len(matches)}")
    run_dir = matches[0]
    metadata_paths = sorted(run_dir.rglob("metadata.json"))
    if len(metadata_paths) != 1:
        raise RuntimeError(f"Expected exactly one metadata file under {run_dir}, found {len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    metadata = json.loads(metadata_path.read_text())
    return run_dir, metadata_path, metadata


def load_test_payload(predictions_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(predictions_path)
    keys = [
        "ic_x",
        "field_x",
        "field_t",
        "test_viscosity",
        "test_initial_conditions",
        "test_targets",
        "test_predictions",
    ]
    return {key: payload[key] for key in keys}


def relative_l2_per_sample(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((prediction - target).reshape(prediction.shape[0], -1), axis=1)
    denominator = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def choose_example_indices(viscosity: np.ndarray) -> tuple[list[int], list[str]]:
    order = np.argsort(viscosity)
    quantiles = [0.1, 0.5, 0.9]
    labels = ["Low viscosity", "Median viscosity", "High viscosity"]
    chosen: list[int] = []
    for quantile in quantiles:
        pos = int(round(quantile * (len(order) - 1)))
        candidate = int(order[pos])
        if candidate in chosen:
            for delta in range(1, len(order)):
                for alt_pos in (max(0, pos - delta), min(len(order) - 1, pos + delta)):
                    alt = int(order[alt_pos])
                    if alt not in chosen:
                        candidate = alt
                        break
                if candidate not in chosen:
                    break
        chosen.append(candidate)
    return chosen, labels


def plot_progress(rows: list[ExperimentRow], out_path: Path) -> dict[str, float]:
    xs = np.arange(1, len(rows) + 1)
    ys = np.asarray([row.val_rel_l2 for row in rows], dtype=np.float64)
    best_so_far = np.minimum.accumulate(ys)
    status_colors = {"keep": "#2563eb", "discard": "#dc2626", "crash": "#6b7280"}

    baseline = rows[0]
    best_row = min(rows, key=lambda row: row.val_rel_l2)
    improvement_factor = baseline.val_rel_l2 / best_row.val_rel_l2
    reduction_pct = 100.0 * (1.0 - best_row.val_rel_l2 / baseline.val_rel_l2)

    fig, ax = plt.subplots(figsize=(11, 6.25))
    ax.plot(xs, ys, color="#94a3b8", lw=1.2, alpha=0.8, zorder=1)
    ax.plot(xs, best_so_far, color="black", lw=2.3, zorder=2, label="best so far")

    for status in ("keep", "discard", "crash"):
        mask = np.asarray([row.status == status for row in rows])
        if np.any(mask):
            ax.scatter(
                xs[mask],
                ys[mask],
                s=58,
                color=status_colors[status],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=status,
            )

    ax.scatter(
        [baseline.index, best_row.index],
        [baseline.val_rel_l2, best_row.val_rel_l2],
        s=90,
        facecolor="none",
        edgecolor="black",
        linewidth=1.6,
        zorder=4,
    )

    ax.annotate(
        f"baseline e{baseline.index}\n{baseline.val_rel_l2:.4e}",
        (baseline.index, baseline.val_rel_l2),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
    )
    ax.annotate(
        f"best e{best_row.index}\n{best_row.val_rel_l2:.4e}",
        (best_row.index, best_row.val_rel_l2),
        textcoords="offset points",
        xytext=(10, -30),
        fontsize=9,
    )

    ax.text(
        0.015,
        0.02,
        (
            f"baseline: {baseline.val_rel_l2:.4e}\n"
            f"best: {best_row.val_rel_l2:.4e}\n"
            f"improvement: {improvement_factor:.2f}x\n"
            f"error reduction: {reduction_pct:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.35"},
    )

    ax.set_title("Large-Run Autoresearch Progress")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Validation Relative L2 Error")
    ax.set_xticks(xs)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_val_rel_l2": baseline.val_rel_l2,
        "best_val_rel_l2": best_row.val_rel_l2,
        "improvement_factor": improvement_factor,
        "error_reduction_pct": reduction_pct,
    }


def plot_field_examples(
    baseline_payload: dict[str, np.ndarray],
    final_payload: dict[str, np.ndarray],
    example_indices: list[int],
    example_labels: list[str],
    out_path: Path,
) -> list[dict[str, float]]:
    ic_x = final_payload["ic_x"]
    field_x = final_payload["field_x"]
    field_t = final_payload["field_t"]
    targets = final_payload["test_targets"]
    baseline_pred = baseline_payload["test_predictions"]
    final_pred = final_payload["test_predictions"]
    initial_conditions = final_payload["test_initial_conditions"]
    viscosity = final_payload["test_viscosity"]

    baseline_errors = relative_l2_per_sample(baseline_pred, targets)
    final_errors = relative_l2_per_sample(final_pred, targets)
    field_limit = max(
        np.max(np.abs(targets[example_indices])),
        np.max(np.abs(baseline_pred[example_indices])),
        np.max(np.abs(final_pred[example_indices])),
    )

    fig, axes = plt.subplots(
        4,
        len(example_indices),
        figsize=(4.6 * len(example_indices), 10.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.8, 1.8, 1.8]},
    )

    if len(example_indices) == 1:
        axes = np.asarray(axes).reshape(4, 1)

    summaries = []
    heatmap = None
    for col, (sample_index, label) in enumerate(zip(example_indices, example_labels, strict=True)):
        ax = axes[0, col]
        ax.plot(ic_x, initial_conditions[sample_index], color="#111827", lw=1.8)
        ax.set_title(f"{label}\n$\\nu$={viscosity[sample_index]:.4f}", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(field_x[0], field_x[-1])
        if col == 0:
            ax.set_ylabel("Input $u(x, 0)$")
        ax.set_xlabel("$x$")

        panels = [
            (axes[1, col], targets[sample_index], "Target field"),
            (axes[2, col], baseline_pred[sample_index], f"Baseline\nrel L2={baseline_errors[sample_index]:.3e}"),
            (axes[3, col], final_pred[sample_index], f"Best run\nrel L2={final_errors[sample_index]:.3e}"),
        ]
        for row_ax, field, title in panels:
            heatmap = row_ax.imshow(
                field,
                origin="lower",
                aspect="auto",
                extent=[field_x[0], field_x[-1], field_t[0], field_t[-1]],
                cmap="coolwarm",
                vmin=-field_limit,
                vmax=field_limit,
            )
            row_ax.set_title(title, fontsize=10)
            row_ax.set_xlabel("$x$")
            if col == 0:
                row_ax.set_ylabel("$t$")

        summaries.append(
            {
                "sample_index": int(sample_index),
                "viscosity": float(viscosity[sample_index]),
                "baseline_test_rel_l2": float(baseline_errors[sample_index]),
                "best_test_rel_l2": float(final_errors[sample_index]),
            }
        )

    if heatmap is None:
        raise RuntimeError("No heatmap created for field examples.")

    fig.colorbar(heatmap, ax=axes[1:, :], shrink=0.9, location="right", label="Field value")
    fig.suptitle("Representative Test Cases: Input to Spatio-Temporal Field", fontsize=14)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return summaries


def plot_slice_examples(
    baseline_payload: dict[str, np.ndarray],
    final_payload: dict[str, np.ndarray],
    example_indices: list[int],
    example_labels: list[str],
    out_path: Path,
) -> None:
    field_x = final_payload["field_x"]
    field_t = final_payload["field_t"]
    targets = final_payload["test_targets"]
    baseline_pred = baseline_payload["test_predictions"]
    final_pred = final_payload["test_predictions"]
    viscosity = final_payload["test_viscosity"]

    slice_targets = [0.25, 0.5, 1.0]
    slice_indices = [int(np.argmin(np.abs(field_t - target_time))) for target_time in slice_targets]

    fig, axes = plt.subplots(
        len(example_indices),
        len(slice_indices),
        figsize=(4.5 * len(slice_indices), 3.3 * len(example_indices) + 0.9),
        sharex=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for col, slice_index in enumerate(slice_indices):
        axes[0, col].set_title(f"$t$={field_t[slice_index]:.2f}", fontsize=11, pad=10)

    for row, (sample_index, label) in enumerate(zip(example_indices, example_labels, strict=True)):
        for col, slice_index in enumerate(slice_indices):
            ax = axes[row, col]
            ax.plot(field_x, targets[sample_index, slice_index], color="black", lw=2.0, label="target")
            ax.plot(field_x, baseline_pred[sample_index, slice_index], color="#ef4444", lw=1.6, ls="--", label="baseline")
            ax.plot(field_x, final_pred[sample_index, slice_index], color="#2563eb", lw=1.6, label="best")
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(f"{label}\n$\\nu$={viscosity[sample_index]:.4f}\n$u(x, t)$")
            if row == len(example_indices) - 1:
                ax.set_xlabel("$x$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("Representative Test Slices: Target vs Baseline vs Best Run", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.84, hspace=0.35, wspace=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = workspace / "results.tsv"
    rows = load_results(results_path)
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 completed experiments, found {len(rows)} in {results_path}")

    bundles = {}
    for row in rows:
        run_dir, metadata_path, metadata = resolve_run_bundle(workspace, row.index)
        metadata_val = float(metadata["eval_metrics"]["val_rel_l2"])
        if abs(metadata_val - row.val_rel_l2) > 5e-8:
            raise RuntimeError(
                f"Metric mismatch for experiment {row.index}: results.tsv={row.val_rel_l2:.9e}, "
                f"metadata={metadata_val:.9e}"
            )
        bundles[row.index] = {
            "row": row,
            "run_dir": run_dir,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "predictions_path": metadata_path.with_name("predictions.npz"),
        }

    baseline_bundle = bundles[1]
    best_row = min(rows, key=lambda row: row.val_rel_l2)
    best_bundle = bundles[best_row.index]

    baseline_payload = load_test_payload(baseline_bundle["predictions_path"])
    best_payload = load_test_payload(best_bundle["predictions_path"])

    example_indices, example_labels = choose_example_indices(best_payload["test_viscosity"])

    progress_stats = plot_progress(rows, output_dir / "figure_progress.svg")
    example_summaries = plot_field_examples(
        baseline_payload,
        best_payload,
        example_indices,
        example_labels,
        output_dir / "figure_examples_fields.svg",
    )
    plot_slice_examples(
        baseline_payload,
        best_payload,
        example_indices,
        example_labels,
        output_dir / "figure_examples_slices.svg",
    )

    summary = {
        "workspace": str(workspace),
        "results_tsv": str(results_path),
        "num_experiments": len(rows),
        "baseline": {
            **asdict(baseline_bundle["row"]),
            "run_dir": str(baseline_bundle["run_dir"]),
            "metadata_path": str(baseline_bundle["metadata_path"]),
            "predictions_path": str(baseline_bundle["predictions_path"]),
            "test_rel_l2": float(baseline_bundle["metadata"]["eval_metrics"]["test_rel_l2"]),
        },
        "best": {
            **asdict(best_bundle["row"]),
            "run_dir": str(best_bundle["run_dir"]),
            "metadata_path": str(best_bundle["metadata_path"]),
            "predictions_path": str(best_bundle["predictions_path"]),
            "test_rel_l2": float(best_bundle["metadata"]["eval_metrics"]["test_rel_l2"]),
        },
        "progress": progress_stats,
        "example_selection": {
            "rule": "test viscosity quantiles at 10%, 50%, and 90%",
            "examples": example_summaries,
        },
        "figures": {
            "progress": str(output_dir / "figure_progress.svg"),
            "fields": str(output_dir / "figure_examples_fields.svg"),
            "slices": str(output_dir / "figure_examples_slices.svg"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
