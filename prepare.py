"""
One-time preparation and fixed utilities for autoresearch Burgers surrogate runs.

Usage:
    python prepare.py --jobs 1
    python prepare.py --force --jobs 32

Artifacts are stored in ~/.cache/autoresearch-burgers/surrogate/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Fixed problem definition (do not modify during experiments)
# ---------------------------------------------------------------------------

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0
TIME_BUDGET = 900  # 15 minutes per experiment on cluster-backed runs

IC_GRID_POINTS = 128
FIELD_X_POINTS = 128
FIELD_T_POINTS = 65

TRAIN_SAMPLES = 512
VAL_SAMPLES = 8
TEST_SAMPLES = 8
DATASET_SEED = 20260322

SOLVER_POINTS = 2049
SOLVER_CFL = 0.2
DATASET_BUILDER = "burgers-surrogate-hgrf-rusanov-ssp-rk3-v1"

VISCOSITY_MIN = 2.5e-3
VISCOSITY_MAX = 5.0e-2
KERNEL_LENGTHSCALE_LOG_MEAN = math.log(0.25)
KERNEL_LENGTHSCALE_LOG_STD = 0.45
KERNEL_LENGTHSCALE_MIN = 0.08
KERNEL_LENGTHSCALE_MAX = 0.70
KERNEL_AMPLITUDE_LOG_MEAN = math.log(0.8)
KERNEL_AMPLITUDE_LOG_STD = 0.35
KERNEL_AMPLITUDE_MIN = 0.25
KERNEL_AMPLITUDE_MAX = 1.25
KERNEL_BIAS_STD = 0.20
KERNEL_JITTER = 1e-6
MAX_INITIAL_ABS = 1.25

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-burgers")
SURROGATE_DIR = os.path.join(CACHE_DIR, "surrogate")
DATASET_FILE = os.path.join(SURROGATE_DIR, "burgers_surrogate_dataset.npz")
MANIFEST_FILE = os.path.join(SURROGATE_DIR, "manifest.json")


@dataclass(frozen=True)
class ProblemSpec:
    x_min: float = X_MIN
    x_max: float = X_MAX
    t_min: float = T_MIN
    t_max: float = T_MAX


@dataclass(frozen=True)
class PriorConfig:
    viscosity_min: float = VISCOSITY_MIN
    viscosity_max: float = VISCOSITY_MAX
    kernel_lengthscale_log_mean: float = KERNEL_LENGTHSCALE_LOG_MEAN
    kernel_lengthscale_log_std: float = KERNEL_LENGTHSCALE_LOG_STD
    kernel_lengthscale_min: float = KERNEL_LENGTHSCALE_MIN
    kernel_lengthscale_max: float = KERNEL_LENGTHSCALE_MAX
    kernel_amplitude_log_mean: float = KERNEL_AMPLITUDE_LOG_MEAN
    kernel_amplitude_log_std: float = KERNEL_AMPLITUDE_LOG_STD
    kernel_amplitude_min: float = KERNEL_AMPLITUDE_MIN
    kernel_amplitude_max: float = KERNEL_AMPLITUDE_MAX
    kernel_bias_std: float = KERNEL_BIAS_STD
    kernel_jitter: float = KERNEL_JITTER
    max_initial_abs: float = MAX_INITIAL_ABS


@dataclass(frozen=True)
class DatasetConfig:
    builder: str = DATASET_BUILDER
    dataset_seed: int = DATASET_SEED
    train_samples: int = TRAIN_SAMPLES
    val_samples: int = VAL_SAMPLES
    test_samples: int = TEST_SAMPLES
    ic_grid_points: int = IC_GRID_POINTS
    field_x_points: int = FIELD_X_POINTS
    field_t_points: int = FIELD_T_POINTS


@dataclass(frozen=True)
class SolverConfig:
    solver_points: int = SOLVER_POINTS
    cfl: float = SOLVER_CFL


PROBLEM_SPEC = ProblemSpec()
PRIOR_CONFIG = PriorConfig()
DATASET_CONFIG = DatasetConfig()
SOLVER_CONFIG = SolverConfig()


# ---------------------------------------------------------------------------
# Fixed grids and priors
# ---------------------------------------------------------------------------

def ic_grid() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, IC_GRID_POINTS, dtype=np.float64)


def field_x_grid() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, FIELD_X_POINTS, dtype=np.float64)


def field_t_grid() -> np.ndarray:
    return np.linspace(T_MIN, T_MAX, FIELD_T_POINTS, dtype=np.float64)


def solver_x_grid() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, SOLVER_POINTS, dtype=np.float64)


def query_coords_grid() -> np.ndarray:
    x_eval = field_x_grid()
    t_eval = field_t_grid()
    grid_x = np.broadcast_to(x_eval[None, :], (FIELD_T_POINTS, FIELD_X_POINTS))
    grid_t = np.broadcast_to(t_eval[:, None], (FIELD_T_POINTS, FIELD_X_POINTS))
    return np.stack((grid_x, grid_t), axis=-1).reshape(-1, 2).astype(np.float32)


def _interior_envelope(x: np.ndarray) -> np.ndarray:
    centered = 2.0 * (x - X_MIN) / (X_MAX - X_MIN) - 1.0
    return np.maximum(1.0 - centered * centered, 0.0)


def _sample_hierarchical_parameters(rng: np.random.Generator) -> dict[str, float]:
    viscosity = float(
        np.exp(
            rng.uniform(
                math.log(PRIOR_CONFIG.viscosity_min),
                math.log(PRIOR_CONFIG.viscosity_max),
            )
        )
    )
    lengthscale = float(
        np.clip(
            np.exp(rng.normal(PRIOR_CONFIG.kernel_lengthscale_log_mean, PRIOR_CONFIG.kernel_lengthscale_log_std)),
            PRIOR_CONFIG.kernel_lengthscale_min,
            PRIOR_CONFIG.kernel_lengthscale_max,
        )
    )
    amplitude = float(
        np.clip(
            np.exp(rng.normal(PRIOR_CONFIG.kernel_amplitude_log_mean, PRIOR_CONFIG.kernel_amplitude_log_std)),
            PRIOR_CONFIG.kernel_amplitude_min,
            PRIOR_CONFIG.kernel_amplitude_max,
        )
    )
    bias = float(rng.normal(0.0, PRIOR_CONFIG.kernel_bias_std))
    return {
        "viscosity": viscosity,
        "kernel_lengthscale": lengthscale,
        "kernel_amplitude": amplitude,
        "kernel_bias": bias,
    }


def _sample_initial_condition(
    x: np.ndarray,
    *,
    kernel_lengthscale: float,
    kernel_amplitude: float,
    kernel_bias: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = x[:, None] - x[None, :]
    covariance = (kernel_amplitude ** 2) * np.exp(-0.5 * (diffs / kernel_lengthscale) ** 2)
    covariance += PRIOR_CONFIG.kernel_jitter * np.eye(x.shape[0], dtype=np.float64)
    chol = np.linalg.cholesky(covariance)
    field = chol @ rng.standard_normal(x.shape[0])
    field = _interior_envelope(x) * (field + kernel_bias)
    max_abs = float(np.max(np.abs(field)))
    if max_abs > PRIOR_CONFIG.max_initial_abs:
        field = field * (PRIOR_CONFIG.max_initial_abs / max_abs)
    return field.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Burgers solver
# ---------------------------------------------------------------------------

def _apply_dirichlet_boundaries(u: np.ndarray) -> np.ndarray:
    u[0] = 0.0
    u[-1] = 0.0
    return u


def _burgers_rhs(u: np.ndarray, *, dx: float, viscosity: float) -> np.ndarray:
    flux_values = 0.5 * u * u
    wave_speeds = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
    flux = 0.5 * (flux_values[:-1] + flux_values[1:]) - 0.5 * wave_speeds * (u[1:] - u[:-1])

    rhs = np.zeros_like(u)
    rhs[1:-1] = (
        -(flux[1:] - flux[:-1]) / dx
        + viscosity * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    )
    return rhs


def _solve_burgers_trajectory(initial_solver: np.ndarray, viscosity: float) -> np.ndarray:
    solver_x = solver_x_grid()
    x_eval = field_x_grid()
    t_eval = field_t_grid()
    dx = float(solver_x[1] - solver_x[0])

    u = _apply_dirichlet_boundaries(initial_solver.astype(np.float64, copy=True))
    u_eval = np.empty((FIELD_T_POINTS, FIELD_X_POINTS), dtype=np.float64)
    u_eval[0] = np.interp(x_eval, solver_x, u)

    current_time = 0.0
    for t_index in range(1, FIELD_T_POINTS):
        target_time = float(t_eval[t_index])
        while current_time < target_time - 1e-15:
            max_speed = max(float(np.max(np.abs(u))), 1e-6)
            dt_conv = SOLVER_CONFIG.cfl * dx / max_speed
            dt_diff = 0.45 * dx * dx / max(viscosity, 1e-12)
            dt = min(target_time - current_time, dt_conv, dt_diff)

            k1 = _burgers_rhs(u, dx=dx, viscosity=viscosity)
            u1 = _apply_dirichlet_boundaries(u + dt * k1)

            k2 = _burgers_rhs(u1, dx=dx, viscosity=viscosity)
            u2 = _apply_dirichlet_boundaries(0.75 * u + 0.25 * (u1 + dt * k2))

            k3 = _burgers_rhs(u2, dx=dx, viscosity=viscosity)
            u = _apply_dirichlet_boundaries((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3))
            current_time += dt

        u_eval[t_index] = np.interp(x_eval, solver_x, u)

    return u_eval.astype(np.float32, copy=False)


def _generate_sample(sample_index: int) -> dict[str, np.ndarray]:
    sample_seed = DATASET_CONFIG.dataset_seed + 7919 * sample_index
    rng = np.random.default_rng(sample_seed)
    params = _sample_hierarchical_parameters(rng)
    x_ic = ic_grid()
    x_solver = solver_x_grid()

    initial_ic = _sample_initial_condition(
        x_ic,
        kernel_lengthscale=params["kernel_lengthscale"],
        kernel_amplitude=params["kernel_amplitude"],
        kernel_bias=params["kernel_bias"],
        seed=sample_seed + 17,
    )
    initial_solver = np.interp(x_solver, x_ic, initial_ic)
    field = _solve_burgers_trajectory(initial_solver, params["viscosity"])

    return {
        "viscosity": np.asarray(params["viscosity"], dtype=np.float32),
        "kernel_lengthscale": np.asarray(params["kernel_lengthscale"], dtype=np.float32),
        "kernel_amplitude": np.asarray(params["kernel_amplitude"], dtype=np.float32),
        "kernel_bias": np.asarray(params["kernel_bias"], dtype=np.float32),
        "initial_condition": initial_ic.astype(np.float32, copy=False),
        "field": field,
    }


def _generate_all_samples(total_samples: int, jobs: int) -> list[dict[str, np.ndarray]]:
    if jobs <= 1:
        return [_generate_sample(index) for index in range(total_samples)]

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_generate_sample, range(total_samples)))


def _split_slice(split: str) -> slice:
    train_stop = DATASET_CONFIG.train_samples
    val_stop = train_stop + DATASET_CONFIG.val_samples
    if split == "train":
        return slice(0, train_stop)
    if split == "val":
        return slice(train_stop, val_stop)
    if split == "test":
        return slice(val_stop, val_stop + DATASET_CONFIG.test_samples)
    raise ValueError(f"Unsupported split: {split}")


def _normalization_payload(train_viscosity: np.ndarray, train_initial: np.ndarray, train_fields: np.ndarray) -> dict[str, np.ndarray]:
    log_viscosity = np.log(train_viscosity.astype(np.float64))
    return {
        "ic_mean": train_initial.mean(axis=0, dtype=np.float64).astype(np.float32),
        "ic_std": (train_initial.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32),
        "field_mean": np.asarray(train_fields.mean(dtype=np.float64), dtype=np.float32),
        "field_std": np.asarray(train_fields.std(dtype=np.float64) + 1e-6, dtype=np.float32),
        "log_viscosity_mean": np.asarray(log_viscosity.mean(dtype=np.float64), dtype=np.float32),
        "log_viscosity_std": np.asarray(log_viscosity.std(dtype=np.float64) + 1e-6, dtype=np.float32),
    }


def _build_dataset(total_samples: int, jobs: int) -> dict[str, np.ndarray]:
    records = _generate_all_samples(total_samples, jobs)
    viscosity = np.asarray([record["viscosity"] for record in records], dtype=np.float32)
    kernel_lengthscale = np.asarray([record["kernel_lengthscale"] for record in records], dtype=np.float32)
    kernel_amplitude = np.asarray([record["kernel_amplitude"] for record in records], dtype=np.float32)
    kernel_bias = np.asarray([record["kernel_bias"] for record in records], dtype=np.float32)
    initial_conditions = np.stack([record["initial_condition"] for record in records], axis=0).astype(np.float32)
    fields = np.stack([record["field"] for record in records], axis=0).astype(np.float32)

    train_slice = _split_slice("train")
    normalization = _normalization_payload(
        viscosity[train_slice],
        initial_conditions[train_slice],
        fields[train_slice],
    )

    payload: dict[str, np.ndarray] = {
        "ic_x": ic_grid().astype(np.float32),
        "field_x": field_x_grid().astype(np.float32),
        "field_t": field_t_grid().astype(np.float32),
        "query_coords": query_coords_grid(),
        **normalization,
    }

    for split in ("train", "val", "test"):
        split_slice = _split_slice(split)
        payload[f"{split}_viscosity"] = viscosity[split_slice]
        payload[f"{split}_kernel_lengthscale"] = kernel_lengthscale[split_slice]
        payload[f"{split}_kernel_amplitude"] = kernel_amplitude[split_slice]
        payload[f"{split}_kernel_bias"] = kernel_bias[split_slice]
        payload[f"{split}_initial_conditions"] = initial_conditions[split_slice]
        payload[f"{split}_fields"] = fields[split_slice]

    return payload


def _manifest_payload(dataset: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "problem": asdict(PROBLEM_SPEC),
        "prior": asdict(PRIOR_CONFIG),
        "dataset": asdict(DATASET_CONFIG),
        "solver": asdict(SOLVER_CONFIG),
        "artifacts": {
            "dataset_file": DATASET_FILE,
            "builder": DATASET_BUILDER,
            "train_samples": int(dataset["train_viscosity"].shape[0]),
            "val_samples": int(dataset["val_viscosity"].shape[0]),
            "test_samples": int(dataset["test_viscosity"].shape[0]),
            "field_shape": list(dataset["train_fields"].shape[1:]),
        },
    }


def _artifact_matches_current_config() -> bool:
    if not (os.path.exists(DATASET_FILE) and os.path.exists(MANIFEST_FILE)):
        return False
    with open(MANIFEST_FILE, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return (
        manifest.get("problem") == asdict(PROBLEM_SPEC)
        and manifest.get("prior") == asdict(PRIOR_CONFIG)
        and manifest.get("dataset") == asdict(DATASET_CONFIG)
        and manifest.get("solver") == asdict(SOLVER_CONFIG)
    )


def _load_dataset_npz() -> dict[str, np.ndarray]:
    with np.load(DATASET_FILE) as data:
        return {name: data[name] for name in data.files}


def ensure_dataset_artifacts(
    *,
    force: bool = False,
    require_existing: bool = False,
    jobs: int = 1,
) -> dict[str, np.ndarray]:
    if _artifact_matches_current_config() and not force:
        return _load_dataset_npz()

    if require_existing:
        raise RuntimeError(
            "Surrogate dataset artifacts are missing or stale. Run `uv run prepare.py --jobs <N>` before training."
        )

    os.makedirs(SURROGATE_DIR, exist_ok=True)
    total_samples = DATASET_CONFIG.train_samples + DATASET_CONFIG.val_samples + DATASET_CONFIG.test_samples
    dataset = _build_dataset(total_samples, jobs=max(1, int(jobs)))
    np.savez(DATASET_FILE, **dataset)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as handle:
        json.dump(_manifest_payload(dataset), handle, indent=2, sort_keys=True)
    return dataset


def load_dataset_artifacts() -> dict[str, np.ndarray]:
    if not _artifact_matches_current_config():
        raise RuntimeError(
            "Surrogate dataset artifacts are missing or stale. Run `uv run prepare.py --jobs <N>` before training."
        )
    return _load_dataset_npz()


def get_split(split: str) -> dict[str, np.ndarray]:
    dataset = load_dataset_artifacts()
    return {
        "viscosity": dataset[f"{split}_viscosity"],
        "initial_conditions": dataset[f"{split}_initial_conditions"],
        "fields": dataset[f"{split}_fields"],
        "kernel_lengthscale": dataset[f"{split}_kernel_lengthscale"],
        "kernel_amplitude": dataset[f"{split}_kernel_amplitude"],
        "kernel_bias": dataset[f"{split}_kernel_bias"],
    }


@jax.jit
def _predict_metrics(
    pred: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    diff = pred - target
    per_sample_rel_l2 = jnp.linalg.norm(diff.reshape((diff.shape[0], -1)), axis=1) / jnp.linalg.norm(
        target.reshape((target.shape[0], -1)),
        axis=1,
    )
    rel_l2 = jnp.mean(per_sample_rel_l2)
    rmse = jnp.sqrt(jnp.mean(diff * diff))
    max_abs = jnp.max(jnp.abs(diff))
    worst_rel_l2 = jnp.max(per_sample_rel_l2)
    return rel_l2, rmse, max_abs, worst_rel_l2


def evaluate_model(
    predict_fields: Callable[[object, jax.Array, jax.Array], jax.Array],
    params: object,
    *,
    split: str = "val",
    batch_size: int = 4,
) -> dict[str, float]:
    split_data = get_split(split)
    viscosity = split_data["viscosity"]
    initial_conditions = split_data["initial_conditions"]
    targets = split_data["fields"]

    preds = []
    for start in range(0, viscosity.shape[0], batch_size):
        stop = min(start + batch_size, viscosity.shape[0])
        batch_viscosity = jnp.asarray(viscosity[start:stop])
        batch_initial = jnp.asarray(initial_conditions[start:stop])
        preds.append(np.asarray(jax.device_get(predict_fields(params, batch_viscosity, batch_initial))))

    pred = jnp.asarray(np.concatenate(preds, axis=0))
    target = jnp.asarray(targets)
    rel_l2, rmse, max_abs, worst_rel_l2 = _predict_metrics(pred, target)
    return {
        "rel_l2": float(rel_l2),
        "rmse": float(rmse),
        "max_abs": float(max_abs),
        "worst_rel_l2": float(worst_rel_l2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fixed Burgers surrogate dataset artifacts for autoresearch")
    parser.add_argument("--force", action="store_true", help="Rebuild the cached dataset even if artifacts already exist")
    parser.add_argument("--jobs", type=int, default=1, help="Number of local worker processes to use while building the dataset")
    args = parser.parse_args()

    t0 = time.time()
    dataset = ensure_dataset_artifacts(force=args.force, require_existing=False, jobs=args.jobs)
    t1 = time.time()

    print(f"Cache directory:  {SURROGATE_DIR}")
    print(f"Dataset file:     {DATASET_FILE}")
    print(f"Manifest file:    {MANIFEST_FILE}")
    print(f"Builder:          {DATASET_BUILDER}")
    print(f"Train samples:    {dataset['train_viscosity'].shape[0]}")
    print(f"Val samples:      {dataset['val_viscosity'].shape[0]}")
    print(f"Test samples:     {dataset['test_viscosity'].shape[0]}")
    print(f"IC grid points:   {dataset['ic_x'].shape[0]}")
    print(f"Field shape:      {dataset['train_fields'].shape[1]} x {dataset['train_fields'].shape[2]}")
    print(f"Wall time:        {t1 - t0:.2f}s")
    print("Done! Ready to train Burgers surrogates.")
