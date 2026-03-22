"""
One-time preparation and fixed utilities for autoresearch Burgers PINN runs.

Usage:
    python prepare.py          # build the fixed reference/evaluation artifacts
    python prepare.py --force  # rebuild even if artifacts already exist

Artifacts are stored in ~/.cache/autoresearch-burgers/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Fixed problem definition (do not modify during experiments)
# ---------------------------------------------------------------------------

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0
VISCOSITY = 0.01 / math.pi
TIME_BUDGET = 300  # 5 minutes, matching autoresearch's fixed-budget setup

# Validation grid used for the fixed metric
EVAL_X_POINTS = 256
EVAL_T_POINTS = 101

# Cole-Hopf / cosine-series reference construction
REFERENCE_SERIES_TERMS = 512
REFERENCE_QUADRATURE_POINTS = 8193
REFERENCE_BUILDER = "cole-hopf-neumann-cosine-series-v1"

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-burgers")
REFERENCE_DIR = os.path.join(CACHE_DIR, "reference")
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "burgers_reference.npz")
MANIFEST_FILE = os.path.join(REFERENCE_DIR, "manifest.json")


@dataclass(frozen=True)
class ProblemSpec:
    x_min: float = X_MIN
    x_max: float = X_MAX
    t_min: float = T_MIN
    t_max: float = T_MAX
    viscosity: float = VISCOSITY


@dataclass(frozen=True)
class ReferenceConfig:
    builder: str = REFERENCE_BUILDER
    series_terms: int = REFERENCE_SERIES_TERMS
    quadrature_points: int = REFERENCE_QUADRATURE_POINTS
    eval_x_points: int = EVAL_X_POINTS
    eval_t_points: int = EVAL_T_POINTS


PROBLEM_SPEC = ProblemSpec()
REFERENCE_CONFIG = ReferenceConfig()


# ---------------------------------------------------------------------------
# Problem helpers
# ---------------------------------------------------------------------------

def initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def boundary_condition(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def _map_unit_interval(samples: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + (upper - lower) * samples


def _sample_unit(num_points: int, dim: int, method: str, seed: int, dtype: np.dtype = np.float32) -> np.ndarray:
    if method == "sobol":
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        power = max(0, math.ceil(math.log2(max(num_points, 1))))
        samples = engine.random_base2(power)[:num_points]
    elif method == "uniform":
        rng = np.random.default_rng(seed)
        samples = rng.random((num_points, dim))
    elif method == "grid":
        side = max(2, math.ceil(num_points ** (1.0 / dim)))
        axes = [np.linspace(0.0, 1.0, side, dtype=np.float64) for _ in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        samples = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)[:num_points]
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    return samples.astype(dtype, copy=False)


def sample_interior_points(
    num_points: int,
    *,
    method: str = "sobol",
    seed: int = 0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    eps_x = 1e-6 * (X_MAX - X_MIN)
    eps_t = 1e-6 * (T_MAX - T_MIN)
    unit = _sample_unit(num_points, dim=2, method=method, seed=seed, dtype=dtype)
    x = _map_unit_interval(unit[:, 0], X_MIN + eps_x, X_MAX - eps_x)
    t = _map_unit_interval(unit[:, 1], T_MIN + eps_t, T_MAX)
    return np.stack((x, t), axis=-1).astype(dtype, copy=False)


def sample_initial_points(
    num_points: int,
    *,
    method: str = "sobol",
    seed: int = 0,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    unit = _sample_unit(num_points, dim=1, method=method, seed=seed, dtype=dtype).squeeze(-1)
    x = _map_unit_interval(unit, X_MIN, X_MAX)
    t = np.full_like(x, T_MIN)
    coords = np.stack((x, t), axis=-1)
    targets = initial_condition(x)[:, None]
    return coords.astype(dtype, copy=False), targets.astype(dtype, copy=False)


def sample_boundary_points(
    num_points: int,
    *,
    method: str = "sobol",
    seed: int = 0,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    left_count = num_points // 2
    right_count = num_points - left_count
    left_unit = _sample_unit(left_count, dim=1, method=method, seed=seed, dtype=dtype).squeeze(-1)
    right_unit = _sample_unit(right_count, dim=1, method=method, seed=seed + 17, dtype=dtype).squeeze(-1)
    t_left = _map_unit_interval(left_unit, T_MIN, T_MAX)
    t_right = _map_unit_interval(right_unit, T_MIN, T_MAX)
    left_coords = np.stack((np.full_like(t_left, X_MIN), t_left), axis=-1)
    right_coords = np.stack((np.full_like(t_right, X_MAX), t_right), axis=-1)
    coords = np.concatenate((left_coords, right_coords), axis=0)
    targets = boundary_condition(coords[:, 1])[:, None]
    return coords.astype(dtype, copy=False), targets.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Reference solution construction
# ---------------------------------------------------------------------------

def _trapz_weights(num_points: int, *, length: float) -> np.ndarray:
    dx = length / (num_points - 1)
    weights = np.full((num_points,), dx, dtype=np.float64)
    weights[0] = 0.5 * dx
    weights[-1] = 0.5 * dx
    return weights


def _phi0(x: np.ndarray) -> np.ndarray:
    exponent = -(np.cos(math.pi * x) + 1.0) / (2.0 * math.pi * VISCOSITY)
    return np.exp(exponent)


def _build_reference_solution(config: ReferenceConfig) -> dict[str, np.ndarray]:
    quad_x = np.linspace(X_MIN, X_MAX, config.quadrature_points, dtype=np.float64)
    quad_weights = _trapz_weights(config.quadrature_points, length=X_MAX - X_MIN)
    phi0 = _phi0(quad_x)

    modes = np.arange(1, config.series_terms + 1, dtype=np.float64)
    theta_quad = 0.5 * math.pi * (quad_x[:, None] + 1.0) * modes[None, :]
    basis_quad = np.cos(theta_quad)

    c0 = 0.5 * np.sum(quad_weights * phi0)
    coeffs = np.sum((quad_weights * phi0)[:, None] * basis_quad, axis=0)
    eigenvalues = (0.5 * math.pi * modes) ** 2

    x_eval = np.linspace(X_MIN, X_MAX, config.eval_x_points, dtype=np.float64)
    t_eval = np.linspace(T_MIN, T_MAX, config.eval_t_points, dtype=np.float64)
    theta_x = 0.5 * math.pi * (x_eval[:, None] + 1.0) * modes[None, :]
    cos_x = np.cos(theta_x)
    sin_x = np.sin(theta_x)
    decay = np.exp(-VISCOSITY * t_eval[:, None] * eigenvalues[None, :])

    weighted = decay[:, None, :] * coeffs[None, None, :]
    phi = c0 + np.sum(weighted * cos_x[None, :, :], axis=-1)
    phi_x = np.sum(
        weighted * (-(0.5 * math.pi) * modes[None, None, :] * sin_x[None, :, :]),
        axis=-1,
    )
    u_eval = -2.0 * VISCOSITY * phi_x / np.clip(phi, 1e-30, None)

    grid_x = np.broadcast_to(x_eval[None, :], (config.eval_t_points, config.eval_x_points))
    grid_t = np.broadcast_to(t_eval[:, None], (config.eval_t_points, config.eval_x_points))
    coords = np.stack((grid_x, grid_t), axis=-1).reshape(-1, 2)
    targets = u_eval.reshape(-1, 1)

    initial_truth = initial_condition(x_eval)
    initial_error = np.max(np.abs(u_eval[0] - initial_truth))
    boundary_error = max(
        np.max(np.abs(u_eval[:, 0])),
        np.max(np.abs(u_eval[:, -1])),
    )

    return {
        "coords": coords.astype(np.float32),
        "targets": targets.astype(np.float32),
        "x_eval": x_eval.astype(np.float32),
        "t_eval": t_eval.astype(np.float32),
        "u_eval": u_eval.astype(np.float32),
        "coeffs": coeffs.astype(np.float32),
        "c0": np.asarray(c0, dtype=np.float64),
        "initial_linf_error": np.asarray(initial_error, dtype=np.float64),
        "boundary_linf_error": np.asarray(boundary_error, dtype=np.float64),
        "series_terms": np.asarray(config.series_terms, dtype=np.int32),
        "quadrature_points": np.asarray(config.quadrature_points, dtype=np.int32),
    }


def _manifest_payload(reference: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "problem": asdict(PROBLEM_SPEC),
        "reference": asdict(REFERENCE_CONFIG),
        "builder": REFERENCE_BUILDER,
        "artifacts": {
            "reference_file": REFERENCE_FILE,
            "coords_points": int(reference["coords"].shape[0]),
            "initial_linf_error": float(reference["initial_linf_error"]),
            "boundary_linf_error": float(reference["boundary_linf_error"]),
        },
    }


def _artifact_matches_current_config() -> bool:
    if not (os.path.exists(REFERENCE_FILE) and os.path.exists(MANIFEST_FILE)):
        return False
    with open(MANIFEST_FILE, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest.get("problem") == asdict(PROBLEM_SPEC) and manifest.get("reference") == asdict(REFERENCE_CONFIG)


def _load_reference_npz() -> dict[str, np.ndarray]:
    with np.load(REFERENCE_FILE) as data:
        return {name: data[name] for name in data.files}


def ensure_reference_artifacts(*, force: bool = False, require_existing: bool = False) -> dict[str, np.ndarray]:
    if _artifact_matches_current_config() and not force:
        return _load_reference_npz()

    if require_existing:
        raise RuntimeError(
            "Reference artifacts are missing or stale. Run `uv run prepare.py` before training."
        )

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    reference = _build_reference_solution(REFERENCE_CONFIG)
    np.savez(REFERENCE_FILE, **reference)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as handle:
        json.dump(_manifest_payload(reference), handle, indent=2, sort_keys=True)
    return reference


def load_reference_artifacts() -> dict[str, np.ndarray]:
    if not _artifact_matches_current_config():
        raise RuntimeError(
            "Reference artifacts are missing or stale. Run `uv run prepare.py` before evaluating."
        )
    return _load_reference_npz()


@jax.jit
def _predict_metrics(
    pred: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    diff = pred - target
    rel_l2 = jnp.linalg.norm(diff) / jnp.linalg.norm(target)
    rmse = jnp.sqrt(jnp.mean(diff * diff))
    max_abs = jnp.max(jnp.abs(diff))
    return rel_l2, rmse, max_abs


def evaluate_model(
    predict_batch: Callable[[object, jax.Array], jax.Array],
    params: object,
    *,
    batch_size: int = 32768,
) -> dict[str, float]:
    reference = load_reference_artifacts()
    coords = reference["coords"]
    targets = reference["targets"]

    preds = []
    for start in range(0, coords.shape[0], batch_size):
        stop = min(start + batch_size, coords.shape[0])
        batch = jnp.asarray(coords[start:stop])
        preds.append(np.asarray(jax.device_get(predict_batch(params, batch))))
    pred = jnp.asarray(np.concatenate(preds, axis=0))
    target = jnp.asarray(targets)
    rel_l2, rmse, max_abs = _predict_metrics(pred, target)
    return {
        "rel_l2": float(rel_l2),
        "rmse": float(rmse),
        "max_abs": float(max_abs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fixed Burgers PINN reference artifacts for autoresearch")
    parser.add_argument("--force", action="store_true", help="Rebuild the cached artifacts even if they already exist")
    args = parser.parse_args()

    t0 = time.time()
    reference = ensure_reference_artifacts(force=args.force, require_existing=False)
    t1 = time.time()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Reference file:  {REFERENCE_FILE}")
    print(f"Manifest file:   {MANIFEST_FILE}")
    print(f"Builder:         {REFERENCE_BUILDER}")
    print(f"Eval points:     {reference['coords'].shape[0]:,}")
    print(f"Initial Linf:    {float(reference['initial_linf_error']):.3e}")
    print(f"Boundary Linf:   {float(reference['boundary_linf_error']):.3e}")
    print(f"Wall time:       {t1 - t0:.2f}s")
    print("Done! Ready to train PINNs on Burgers.")
