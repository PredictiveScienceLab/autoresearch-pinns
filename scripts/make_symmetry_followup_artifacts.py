#!/usr/bin/env python3
"""
Generate a continued progress figure and structure summary for the symmetry run.

Usage:
    python3 scripts/make_symmetry_followup_artifacts.py \
        --large-workspace /tmp/autoresearch-codex-large.woja6k \
        --symmetry-workspace /tmp/autoresearch-codex-symmetry.WxRZKw \
        --output-dir artifacts/2026-03-24-symmetry-run
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
    campaign: str
    campaign_index: int
    global_index: int
    commit: str
    val_rel_l2: float
    memory_gb: float
    status: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large-workspace",
        type=Path,
        default=Path("/tmp/autoresearch-codex-large.woja6k"),
        help="Path to the large-run isolated workspace.",
    )
    parser.add_argument(
        "--symmetry-workspace",
        type=Path,
        default=Path("/tmp/autoresearch-codex-symmetry.WxRZKw"),
        help="Path to the symmetry-run isolated workspace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/2026-03-24-symmetry-run"),
        help="Directory to write the figure and summaries.",
    )
    return parser.parse_args()


def load_results(path: Path, campaign: str, *, start_index: int) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            rows.append(
                ExperimentRow(
                    campaign=campaign,
                    campaign_index=idx,
                    global_index=start_index + idx - 1,
                    commit=row["commit"],
                    val_rel_l2=float(row["val_rel_l2"]),
                    memory_gb=float(row["memory_gb"]),
                    status=row["status"],
                    description=row["description"],
                )
            )
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def best_row(rows: list[ExperimentRow]) -> ExperimentRow:
    valid = [row for row in rows if row.status != "crash" and row.val_rel_l2 > 0.0]
    if not valid:
        raise RuntimeError("No valid non-crash rows found")
    return min(valid, key=lambda row: row.val_rel_l2)


def plot_value(rows: list[ExperimentRow], row: ExperimentRow) -> float:
    positive = [item.val_rel_l2 for item in rows if item.val_rel_l2 > 0.0]
    max_positive = max(positive)
    if row.val_rel_l2 <= 0.0:
        return max_positive * 1.5
    return row.val_rel_l2


def running_best(rows: list[ExperimentRow]) -> np.ndarray:
    current = np.inf
    values = []
    for row in rows:
        if row.val_rel_l2 > 0.0 and row.status != "crash":
            current = min(current, row.val_rel_l2)
        values.append(current)
    return np.asarray(values, dtype=np.float64)


def hidden_reason(row: ExperimentRow, focus_cutoff: float) -> str | None:
    if row.status == "crash":
        return "crashed"
    if row.val_rel_l2 <= 0.0:
        return "crashed"
    if row.val_rel_l2 > focus_cutoff:
        return "error too high"
    return None


def plot_progress(
    large_rows: list[ExperimentRow],
    symmetry_rows: list[ExperimentRow],
    out_svg: Path,
    out_png: Path,
) -> dict[str, float]:
    all_rows = large_rows + symmetry_rows
    status_colors = {"keep": "#2563eb", "discard": "#dc2626", "crash": "#6b7280"}

    large_best = best_row(large_rows)
    symmetry_baseline = symmetry_rows[0]
    symmetry_best = best_row(symmetry_rows)
    overall_baseline = large_rows[0]
    focus_cutoff = overall_baseline.val_rel_l2 * 1.05
    visible_rows = [
        row
        for row in all_rows
        if row.status != "crash" and row.val_rel_l2 > 0.0 and row.val_rel_l2 <= focus_cutoff
    ]
    hidden_rows = [(row, hidden_reason(row, focus_cutoff)) for row in all_rows]
    hidden_rows = [(row, reason) for row, reason in hidden_rows if reason is not None]
    xs = np.asarray([row.global_index for row in visible_rows], dtype=np.int32)
    ys = np.asarray([row.val_rel_l2 for row in visible_rows], dtype=np.float64)
    best_so_far = running_best(visible_rows)

    overall_improvement = overall_baseline.val_rel_l2 / symmetry_best.val_rel_l2
    overall_reduction_pct = 100.0 * (1.0 - symmetry_best.val_rel_l2 / overall_baseline.val_rel_l2)
    symmetry_improvement = symmetry_baseline.val_rel_l2 / symmetry_best.val_rel_l2
    symmetry_reduction_pct = 100.0 * (1.0 - symmetry_best.val_rel_l2 / symmetry_baseline.val_rel_l2)

    fig, (ax_hidden, ax) = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 5.0], "hspace": 0.05},
    )
    for axis in (ax_hidden, ax):
        axis.axvspan(0.5, len(large_rows) + 0.5, facecolor="#f3f4f6", alpha=0.65, zorder=0)
        axis.axvspan(len(large_rows) + 0.5, len(all_rows) + 0.5, facecolor="#eff6ff", alpha=0.65, zorder=0)
        axis.axvline(len(large_rows) + 0.5, color="#111827", lw=1.1, ls="--", alpha=0.9, zorder=1)

    hidden_y = {"error too high": 0, "crashed": 1}
    hidden_colors = {"error too high": "#f59e0b", "crashed": "#6b7280"}
    hidden_markers = {"error too high": "v", "crashed": "X"}
    for reason in ("error too high", "crashed"):
        reason_rows = [row for row, row_reason in hidden_rows if row_reason == reason]
        if reason_rows:
            reason_x = [row.global_index for row in reason_rows]
            reason_y = [hidden_y[reason]] * len(reason_rows)
            ax_hidden.scatter(
                reason_x,
                reason_y,
                s=72,
                color=hidden_colors[reason],
                marker=hidden_markers[reason],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=reason,
            )

    ax_hidden.set_yticks([0, 1], labels=["error too high", "crashed"])
    ax_hidden.set_ylim(-0.6, 1.6)
    ax_hidden.grid(True, axis="x", alpha=0.18)
    ax_hidden.grid(False, axis="y")
    ax_hidden.tick_params(axis="x", labelbottom=False)
    ax_hidden.text(
        0.01,
        0.98,
        "Hidden-from-zoom strip",
        transform=ax_hidden.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    if hidden_rows:
        ax_hidden.legend(frameon=False, ncol=2, loc="upper right")

    ax.plot(xs, ys, color="#9ca3af", lw=1.1, alpha=0.9, zorder=1)
    ax.plot(xs, best_so_far, color="black", lw=2.2, zorder=2, label="best so far")

    for status in ("keep", "discard"):
        mask = np.asarray([row.status == status for row in visible_rows])
        if np.any(mask):
            ax.scatter(
                xs[mask],
                ys[mask],
                s=56,
                color=status_colors[status],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=status,
            )

    for row, label, dy in (
        (overall_baseline, f"overall baseline e{overall_baseline.global_index}", 10),
        (large_best, f"large best e{large_best.global_index}", -28),
        (symmetry_baseline, f"symmetry baseline e{symmetry_baseline.global_index}", 14),
        (symmetry_best, f"symmetry best e{symmetry_best.global_index}", -26),
    ):
        ax.scatter(
            [row.global_index],
            [row.val_rel_l2],
            s=96,
            facecolor="none",
            edgecolor="black",
            linewidth=1.4,
            zorder=4,
        )
        ax.annotate(
            f"{label}\n{row.val_rel_l2:.4e}",
            (row.global_index, row.val_rel_l2),
            textcoords="offset points",
            xytext=(8, dy),
            fontsize=8.8,
        )

    ax.text(
        0.18,
        0.98,
        "Large operator search",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.78,
        0.98,
        "Symmetry search",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.015,
        0.02,
        (
            f"overall: {overall_baseline.val_rel_l2:.4e} -> {symmetry_best.val_rel_l2:.4e} "
            f"({overall_improvement:.2f}x, {overall_reduction_pct:.1f}% lower)\n"
            f"symmetry stage: {symmetry_baseline.val_rel_l2:.4e} -> {symmetry_best.val_rel_l2:.4e} "
            f"({symmetry_improvement:.2f}x, {symmetry_reduction_pct:.1f}% lower)\n"
            f"focused view: non-crash runs with val_rel_l2 <= {focus_cutoff:.4e}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.35"},
    )

    ax.text(
        0.985,
        0.02,
        "Every run is shown: the top strip marks `crashed` and `error too high` cases,\n"
        "while the main panel zooms in on the competitive error range.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )

    xticks = np.arange(1, len(all_rows) + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, len(all_rows) + 0.5)
    ax.set_xlabel("Experiment (1–20 large run, 21–40 symmetry run)")
    ax.set_ylabel("Validation Relative L2 Error")
    span = float(ys.max() - ys.min())
    pad = max(span * 0.12, 4e-4)
    ax.set_ylim(max(0.0, float(ys.min()) - pad), float(ys.max()) + pad)
    ax.set_title("Autoresearch Progress Continued Through Symmetry Search (Focused View)")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "overall_baseline_val_rel_l2": overall_baseline.val_rel_l2,
        "large_best_val_rel_l2": large_best.val_rel_l2,
        "symmetry_baseline_val_rel_l2": symmetry_baseline.val_rel_l2,
        "symmetry_best_val_rel_l2": symmetry_best.val_rel_l2,
        "overall_improvement_factor": overall_improvement,
        "overall_error_reduction_pct": overall_reduction_pct,
        "symmetry_improvement_factor": symmetry_improvement,
        "symmetry_error_reduction_pct": symmetry_reduction_pct,
        "symmetry_best_global_index": symmetry_best.global_index,
        "focused_cutoff_val_rel_l2": focus_cutoff,
        "focused_visible_rows": len(visible_rows),
        "hidden_rows": len(hidden_rows),
        "hidden_error_too_high_rows": sum(1 for _, reason in hidden_rows if reason == "error too high"),
        "hidden_crashed_rows": sum(1 for _, reason in hidden_rows if reason == "crashed"),
    }


def resolve_metadata_by_val(checkpoint_root: Path, target: float, *, tolerance: float = 5e-7) -> tuple[Path, dict]:
    matches = []
    for meta_path in checkpoint_root.glob("*/metadata.json"):
        meta = json.loads(meta_path.read_text())
        value = meta.get("eval_metrics", {}).get("val_rel_l2")
        if value is None:
            continue
        delta = abs(float(value) - target)
        if delta <= tolerance:
            matches.append((delta, meta_path, meta))
    if not matches:
        raise RuntimeError(f"Could not resolve metadata for val_rel_l2={target:.9e}")
    matches.sort(key=lambda item: item[0])
    return matches[0][1], matches[0][2]


def normalized_model_config(meta: dict) -> dict[str, object]:
    config = dict(meta["model_config"])
    config.setdefault("branch_input_mode", "raw")
    config.setdefault("symmetry_mode", "none")
    config.setdefault("boundary_mode", "none")
    return config


def structure_summary(
    baseline_meta: dict,
    best_meta: dict,
    out_md: Path,
    out_json: Path,
) -> None:
    baseline_cfg = normalized_model_config(baseline_meta)
    best_cfg = normalized_model_config(best_meta)
    param_delta = int(best_meta["num_params"]) - int(baseline_meta["num_params"])
    param_delta_pct = 100.0 * param_delta / float(baseline_meta["num_params"])

    summary = {
        "symmetry_baseline": {
            "val_rel_l2": baseline_meta["eval_metrics"]["val_rel_l2"],
            "num_params": baseline_meta["num_params"],
            "num_params_m": baseline_meta["num_params_m"],
            "model_config": baseline_cfg,
        },
        "symmetry_best": {
            "val_rel_l2": best_meta["eval_metrics"]["val_rel_l2"],
            "num_params": best_meta["num_params"],
            "num_params_m": best_meta["num_params_m"],
            "model_config": best_cfg,
        },
        "parameter_delta": {
            "absolute": param_delta,
            "percent": param_delta_pct,
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    unchanged = [
        f"branch/trunk families stayed `{baseline_cfg['branch_family']}` / `{baseline_cfg['trunk_family']}`",
        f"hidden widths stayed `{baseline_cfg['branch_hidden_dim']}` / `{baseline_cfg['trunk_hidden_dim']}`",
        f"latent size stayed `{baseline_cfg['latent_dim']}`",
        f"Fourier feature count and scale stayed `{baseline_cfg['fourier_features']}` and `{baseline_cfg['fourier_scale']}`",
    ]
    changed = [
        f"coordinate encoding: `{baseline_cfg['coord_encoding']}` -> `{best_cfg['coord_encoding']}`",
        f"branch input mode: `{baseline_cfg['branch_input_mode']}` -> `{best_cfg['branch_input_mode']}`",
        f"symmetry mode: `{baseline_cfg['symmetry_mode']}` -> `{best_cfg['symmetry_mode']}`",
        f"boundary mode: `{baseline_cfg['boundary_mode']}` -> `{best_cfg['boundary_mode']}`",
        (
            f"parameter count: `{baseline_meta['num_params_m']:.3f}M` -> "
            f"`{best_meta['num_params_m']:.3f}M` (`+{param_delta:,}` params, `{param_delta_pct:.1f}%`)"
        ),
    ]

    lines = [
        "# Symmetry Run Structure Change",
        "",
        "The symmetry run did not win by scaling the network again. It kept the same DeepONet skeleton and mostly changed the representation so the known finite-domain symmetry is built in.",
        "",
        f"Baseline control (`26ecc84`, `val_rel_l2 = {baseline_meta['eval_metrics']['val_rel_l2']:.7e}`):",
        f"- `DeepONet` with `{baseline_cfg['branch_family']}` branch / `{baseline_cfg['trunk_family']}` trunk",
        f"- hidden widths `{baseline_cfg['branch_hidden_dim']}` / `{baseline_cfg['trunk_hidden_dim']}`, latent `{baseline_cfg['latent_dim']}`",
        f"- generic `{baseline_cfg['coord_encoding']}` trunk coordinates, raw initial-condition branch input",
        f"- no exact symmetry projection and no boundary envelope",
        "",
        f"Best kept symmetry model (`d4f60d6`, `val_rel_l2 = {best_meta['eval_metrics']['val_rel_l2']:.7e}`):",
        f"- same `DeepONet` branch/trunk families and the same hidden widths / latent size",
        f"- parity-aware `{best_cfg['coord_encoding']}` trunk coordinates",
        f"- `{best_cfg['branch_input_mode']}` branch input so the model sees the mirror-sign partner explicitly",
        f"- exact `{best_cfg['symmetry_mode']}` at prediction time",
        f"- `{best_cfg['boundary_mode']}` envelope so the output respects the bounded Dirichlet walls",
        "",
        "What changed structurally:",
    ]
    lines.extend(f"- {line}" for line in changed)
    lines.append("")
    lines.append("What stayed fixed:")
    lines.extend(f"- {line}" for line in unchanged)
    lines.append("")
    lines.append(
        "Short read: the improvement came from imposing the correct bounded-domain mirror-sign structure, not from making the network much larger."
    )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    large_rows = load_results(args.large_workspace / "results.tsv", "large", start_index=1)
    symmetry_rows = load_results(
        args.symmetry_workspace / "results.tsv",
        "symmetry",
        start_index=len(large_rows) + 1,
    )

    metrics = plot_progress(
        large_rows,
        symmetry_rows,
        args.output_dir / "figure_progress_continued.svg",
        args.output_dir / "figure_progress_continued.png",
    )

    symmetry_checkpoint_root = args.symmetry_workspace / "results" / "checkpoints"
    baseline_row = symmetry_rows[0]
    best_symmetry_row = best_row(symmetry_rows)
    _, baseline_meta = resolve_metadata_by_val(symmetry_checkpoint_root, baseline_row.val_rel_l2)
    _, best_meta = resolve_metadata_by_val(symmetry_checkpoint_root, best_symmetry_row.val_rel_l2)
    structure_summary(
        baseline_meta,
        best_meta,
        args.output_dir / "structure_change_summary.md",
        args.output_dir / "summary.json",
    )

    # Keep the exact ledgers alongside the figure for traceability.
    (args.output_dir / "results_large.tsv").write_text((args.large_workspace / "results.tsv").read_text())
    (args.output_dir / "results_symmetry.tsv").write_text((args.symmetry_workspace / "results.tsv").read_text())

    merged_summary = {
        "large_rows": len(large_rows),
        "symmetry_rows": len(symmetry_rows),
        **metrics,
    }
    (args.output_dir / "progress_summary.json").write_text(json.dumps(merged_summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
