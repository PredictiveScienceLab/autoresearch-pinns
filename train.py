"""
Autoresearch PINN training script for the 1D viscous Burgers equation.

Usage:
    uv run train.py
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from prepare import (
    TIME_BUDGET,
    VISCOSITY,
    ensure_reference_artifacts,
    evaluate_model,
    sample_boundary_points,
    sample_initial_points,
    sample_interior_points,
)

jax.config.update("jax_default_matmul_precision", "high")


# ---------------------------------------------------------------------------
# Editable experiment surface
# ---------------------------------------------------------------------------

SEED = 1337

# Network representation
NETWORK_FAMILY = "mlp"      # mlp | resmlp | siren
INPUT_ENCODING = "raw"      # raw | fourier
ACTIVATION = "tanh"         # tanh | silu | gelu | relu
DEPTH = 6                   # hidden depth / number of blocks
HIDDEN_DIM = 128
FOURIER_FEATURES = 64       # ignored unless INPUT_ENCODING == "fourier"
FOURIER_SCALE = 4.0
FOURIER_SEED = 0
SIREN_FIRST_W0 = 30.0
SIREN_HIDDEN_W0 = 1.0

# Training data and loss weighting
INTERIOR_POINTS = 1024
INITIAL_POINTS = 256
BOUNDARY_POINTS = 256
SAMPLING_METHOD = "sobol"   # sobol | uniform | grid
SAMPLING_POOL_MULTIPLIER = 64
INTERIOR_WEIGHT = 1.0
INITIAL_WEIGHT = 25.0
BOUNDARY_WEIGHT = 25.0

# Optimization
GRAD_CLIP_NORM = 1.0
EVAL_BATCH_SIZE = 32768
SCHEDULER_NAME = "cosine"   # constant | cosine
WARMUP_FRACTION = 0.05
MIN_LR_RATIO = 0.2


@dataclass(frozen=True)
class OptimizerPhase:
    name: str
    kind: str
    duration_ratio: float
    lr: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    decay: float = 0.9
    nesterov: bool = False


OPTIMIZER_PHASES = (
    OptimizerPhase(
        name="adam-baseline",
        kind="adam",
        duration_ratio=1.0,
        lr=1e-3,
        weight_decay=0.0,
    ),
)


# ---------------------------------------------------------------------------
# Equinox model definitions
# ---------------------------------------------------------------------------

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0
X_DIRECTION = jnp.array([1.0, 0.0], dtype=jnp.float32)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    input_encoding: str
    activation: str
    depth: int
    hidden_dim: int
    fourier_features: int
    fourier_scale: float
    fourier_seed: int
    siren_first_w0: float
    siren_hidden_w0: float


def make_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    table = {
        "tanh": jnp.tanh,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def override_linear_params(linear: eqx.nn.Linear, weight: jax.Array, bias: jax.Array) -> eqx.nn.Linear:
    linear = eqx.tree_at(lambda layer: layer.weight, linear, weight)
    linear = eqx.tree_at(lambda layer: layer.bias, linear, bias)
    return linear


def make_siren_linear(in_dim: int, out_dim: int, *, key: jax.Array, w0: float, is_first: bool) -> eqx.nn.Linear:
    w_key, b_key, base_key = jax.random.split(key, 3)
    if is_first:
        bound = 1.0 / in_dim
    else:
        bound = math.sqrt(6.0 / in_dim) / w0
    weight = jax.random.uniform(w_key, (out_dim, in_dim), minval=-bound, maxval=bound, dtype=jnp.float32)
    bias = jax.random.uniform(b_key, (out_dim,), minval=-bound, maxval=bound, dtype=jnp.float32)
    linear = eqx.nn.Linear(in_dim, out_dim, key=base_key, use_bias=True)
    return override_linear_params(linear, weight, bias)


class CoordinateEncoder(eqx.Module):
    input_encoding: str = eqx.field(static=True)
    offset: tuple[float, float] = eqx.field(static=True)
    scale: tuple[float, float] = eqx.field(static=True)
    encoder_matrix: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def __init__(self, input_encoding: str, *, fourier_features: int, fourier_scale: float, fourier_seed: int):
        self.input_encoding = input_encoding
        self.offset = (float(X_MIN), float(T_MIN))
        self.scale = (float(X_MAX - X_MIN), float(T_MAX - T_MIN))
        if input_encoding == "fourier":
            rng = np.random.default_rng(fourier_seed)
            matrix = (rng.standard_normal((2, fourier_features)) * fourier_scale).astype(np.float32)
            self.encoder_matrix = tuple(tuple(float(value) for value in row) for row in matrix)
        else:
            self.encoder_matrix = None

    def __call__(self, coords: jax.Array) -> jax.Array:
        offset = jnp.asarray(self.offset, dtype=jnp.float32)
        scale = jnp.asarray(self.scale, dtype=jnp.float32)
        coords = 2.0 * (coords - offset) / scale - 1.0
        if self.input_encoding == "fourier":
            encoder_matrix = jnp.asarray(self.encoder_matrix, dtype=jnp.float32)
            projected = 2.0 * math.pi * (coords @ encoder_matrix)
            coords = jnp.concatenate((coords, jnp.sin(projected), jnp.cos(projected)), axis=-1)
        return coords


class PlainMLP(eqx.Module):
    encoder: CoordinateEncoder
    net: eqx.nn.MLP

    def __init__(self, config: ModelConfig, *, key: jax.Array):
        encoded_dim = 2 + 2 * config.fourier_features if config.input_encoding == "fourier" else 2
        self.encoder = CoordinateEncoder(
            config.input_encoding,
            fourier_features=config.fourier_features,
            fourier_scale=config.fourier_scale,
            fourier_seed=config.fourier_seed,
        )
        self.net = eqx.nn.MLP(
            in_size=encoded_dim,
            out_size=1,
            width_size=config.hidden_dim,
            depth=config.depth,
            activation=make_activation(config.activation),
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, coords: jax.Array) -> jax.Array:
        return jnp.squeeze(self.net(self.encoder(coords)), axis=-1)


class ResidualBlock(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    activation_name: str = eqx.field(static=True)

    def __init__(self, hidden_dim: int, activation_name: str, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.activation_name = activation_name

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = make_activation(self.activation_name)
        residual = x
        x = activation(self.fc1(x))
        x = self.fc2(x)
        return activation(residual + x)


class ResidualMLP(eqx.Module):
    encoder: CoordinateEncoder
    stem: eqx.nn.Linear
    blocks: tuple[ResidualBlock, ...]
    head: eqx.nn.Linear
    activation_name: str = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key: jax.Array):
        encoded_dim = 2 + 2 * config.fourier_features if config.input_encoding == "fourier" else 2
        self.encoder = CoordinateEncoder(
            config.input_encoding,
            fourier_features=config.fourier_features,
            fourier_scale=config.fourier_scale,
            fourier_seed=config.fourier_seed,
        )
        keys = jax.random.split(key, config.depth + 2)
        self.stem = eqx.nn.Linear(encoded_dim, config.hidden_dim, key=keys[0])
        self.blocks = tuple(
            ResidualBlock(config.hidden_dim, config.activation, key=block_key)
            for block_key in keys[1:-1]
        )
        self.head = eqx.nn.Linear(config.hidden_dim, 1, key=keys[-1])
        self.activation_name = config.activation

    def __call__(self, coords: jax.Array) -> jax.Array:
        activation = make_activation(self.activation_name)
        x = activation(self.stem(self.encoder(coords)))
        for block in self.blocks:
            x = block(x)
        return jnp.squeeze(self.head(x), axis=-1)


class SirenLayer(eqx.Module):
    linear: eqx.nn.Linear
    w0: float = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array, w0: float, is_first: bool):
        self.linear = make_siren_linear(in_dim, out_dim, key=key, w0=w0, is_first=is_first)
        self.w0 = w0

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sin(self.w0 * self.linear(x))


class Siren(eqx.Module):
    encoder: CoordinateEncoder
    layers: tuple[SirenLayer, ...]
    head: eqx.nn.Linear
    hidden_w0: float = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key: jax.Array):
        if config.input_encoding != "raw":
            raise ValueError("SIREN expects raw coordinates; disable Fourier features for this family.")
        self.encoder = CoordinateEncoder(
            "raw",
            fourier_features=0,
            fourier_scale=1.0,
            fourier_seed=0,
        )
        keys = jax.random.split(key, config.depth + 1)
        layers = [
            SirenLayer(2, config.hidden_dim, key=keys[0], w0=config.siren_first_w0, is_first=True)
        ]
        for layer_key in keys[1:]:
            layers.append(
                SirenLayer(
                    config.hidden_dim,
                    config.hidden_dim,
                    key=layer_key,
                    w0=config.siren_hidden_w0,
                    is_first=False,
                )
            )
        head_key = jax.random.fold_in(key, 17)
        self.head = make_siren_linear(
            config.hidden_dim,
            1,
            key=head_key,
            w0=config.siren_hidden_w0,
            is_first=False,
        )
        self.layers = tuple(layers)
        self.hidden_w0 = config.siren_hidden_w0

    def __call__(self, coords: jax.Array) -> jax.Array:
        x = self.encoder(coords)
        for layer in self.layers:
            x = layer(x)
        return jnp.squeeze(self.head(x), axis=-1)


def build_model(config: ModelConfig, *, key: jax.Array) -> eqx.Module:
    if config.family == "mlp":
        return PlainMLP(config, key=key)
    if config.family == "resmlp":
        return ResidualMLP(config, key=key)
    if config.family == "siren":
        return Siren(config, key=key)
    raise ValueError(f"Unsupported network family: {config.family}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingPools:
    interior_pool: jax.Array
    interior_size: int
    initial_coords_pool: jax.Array
    initial_targets_pool: jax.Array
    initial_size: int
    boundary_coords_pool: jax.Array
    boundary_targets_pool: jax.Array
    boundary_size: int


@dataclass(frozen=True)
class PhaseRuntime:
    phase: OptimizerPhase
    optimizer: optax.GradientTransformation
    step_fn: Callable[[eqx.Module, optax.OptState, jax.Array, jax.Array], tuple[eqx.Module, optax.OptState, dict[str, jax.Array]]]


def count_parameters(model: eqx.Module) -> int:
    arrays = eqx.filter(model, eqx.is_inexact_array)
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(arrays)))


def build_optimizer(phase: OptimizerPhase) -> optax.GradientTransformation:
    kind = phase.kind.lower()
    transforms = []
    if GRAD_CLIP_NORM is not None:
        transforms.append(optax.clip_by_global_norm(GRAD_CLIP_NORM))
    if kind == "adam":
        transforms.append(optax.adam(learning_rate=1.0, b1=phase.betas[0], b2=phase.betas[1], eps=phase.eps))
    elif kind == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=1.0,
                b1=phase.betas[0],
                b2=phase.betas[1],
                eps=phase.eps,
                weight_decay=phase.weight_decay,
            )
        )
    elif kind == "rmsprop":
        if phase.weight_decay != 0.0:
            transforms.append(optax.add_decayed_weights(phase.weight_decay))
        transforms.append(
            optax.rmsprop(
                learning_rate=1.0,
                decay=phase.decay,
                eps=phase.eps,
                momentum=phase.momentum,
            )
        )
    elif kind == "sgd":
        if phase.weight_decay != 0.0:
            transforms.append(optax.add_decayed_weights(phase.weight_decay))
        transforms.append(optax.sgd(learning_rate=1.0, momentum=phase.momentum, nesterov=phase.nesterov))
    else:
        raise ValueError(f"Unsupported optimizer kind: {phase.kind}")
    return optax.chain(*transforms)


def phase_schedule_multiplier(phase_progress: float) -> float:
    progress = max(0.0, min(1.0, phase_progress))
    if SCHEDULER_NAME == "constant":
        return 1.0
    if SCHEDULER_NAME != "cosine":
        raise ValueError(f"Unsupported scheduler: {SCHEDULER_NAME}")
    if WARMUP_FRACTION > 0.0 and progress < WARMUP_FRACTION:
        return progress / WARMUP_FRACTION
    if WARMUP_FRACTION >= 1.0:
        cosine_progress = 0.0
    else:
        cosine_progress = (progress - WARMUP_FRACTION) / max(1.0 - WARMUP_FRACTION, 1e-8)
        cosine_progress = max(0.0, min(1.0, cosine_progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def validate_phases(phases: tuple[OptimizerPhase, ...]) -> tuple[float, ...]:
    if not phases:
        raise ValueError("At least one optimizer phase is required.")
    ratios = [phase.duration_ratio for phase in phases]
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("Optimizer phase duration_ratio values must be positive.")
    total = sum(ratios)
    return tuple(ratio / total for ratio in ratios)


def _double_pool(array: np.ndarray) -> jax.Array:
    device_array = jnp.asarray(array)
    return jnp.concatenate((device_array, device_array), axis=0)


def build_sampling_pools() -> SamplingPools:
    interior_size = max(INTERIOR_POINTS, INTERIOR_POINTS * SAMPLING_POOL_MULTIPLIER)
    initial_size = max(INITIAL_POINTS, INITIAL_POINTS * SAMPLING_POOL_MULTIPLIER)
    boundary_size = max(BOUNDARY_POINTS, BOUNDARY_POINTS * SAMPLING_POOL_MULTIPLIER)

    interior_pool = sample_interior_points(interior_size, method=SAMPLING_METHOD, seed=SEED + 101)
    initial_coords, initial_targets = sample_initial_points(initial_size, method=SAMPLING_METHOD, seed=SEED + 202)
    boundary_coords, boundary_targets = sample_boundary_points(boundary_size, method=SAMPLING_METHOD, seed=SEED + 303)

    return SamplingPools(
        interior_pool=_double_pool(interior_pool),
        interior_size=interior_size,
        initial_coords_pool=_double_pool(initial_coords),
        initial_targets_pool=_double_pool(initial_targets),
        initial_size=initial_size,
        boundary_coords_pool=_double_pool(boundary_coords),
        boundary_targets_pool=_double_pool(boundary_targets),
        boundary_size=boundary_size,
    )


def _slice_pool(pool: jax.Array, pool_size: int, batch_size: int, start: jax.Array) -> jax.Array:
    start = jnp.mod(start, pool_size)
    return jax.lax.dynamic_slice_in_dim(pool, start, batch_size, axis=0)


def scale_updates(updates: Any, learning_rate: jax.Array) -> Any:
    return jax.tree_util.tree_map(
        lambda update: None if update is None else learning_rate * update,
        updates,
    )


def make_train_step(
    pools: SamplingPools,
    optimizer: optax.GradientTransformation,
) -> Callable[[eqx.Module, optax.OptState, jax.Array, jax.Array], tuple[eqx.Module, optax.OptState, dict[str, jax.Array]]]:
    def sample_batch(step_index: jax.Array) -> dict[str, jax.Array]:
        interior_start = step_index * INTERIOR_POINTS
        initial_start = step_index * (INITIAL_POINTS * 3)
        boundary_start = step_index * (BOUNDARY_POINTS * 5)
        return {
            "interior": _slice_pool(pools.interior_pool, pools.interior_size, INTERIOR_POINTS, interior_start),
            "initial_coords": _slice_pool(
                pools.initial_coords_pool,
                pools.initial_size,
                INITIAL_POINTS,
                initial_start,
            ),
            "initial_targets": _slice_pool(
                pools.initial_targets_pool,
                pools.initial_size,
                INITIAL_POINTS,
                initial_start,
            ),
            "boundary_coords": _slice_pool(
                pools.boundary_coords_pool,
                pools.boundary_size,
                BOUNDARY_POINTS,
                boundary_start,
            ),
            "boundary_targets": _slice_pool(
                pools.boundary_targets_pool,
                pools.boundary_size,
                BOUNDARY_POINTS,
                boundary_start,
            ),
        }

    def scalar_field(model: eqx.Module, point: jax.Array) -> jax.Array:
        return model(point)

    def point_quantities(model: eqx.Module, point: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        value_and_grad = jax.value_and_grad(lambda current_point: scalar_field(model, current_point))
        (u, grad), (_, hessian_x_col) = jax.jvp(value_and_grad, (point,), (X_DIRECTION,))
        return u, grad[0], grad[1], hessian_x_col[0]

    batched_point_quantities = jax.vmap(point_quantities, in_axes=(None, 0))
    batched_predict = jax.vmap(lambda model, point: model(point), in_axes=(None, 0))

    def compute_losses(model: eqx.Module, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        u, u_x, u_t, u_xx = batched_point_quantities(model, batch["interior"])
        residual = u_t + u * u_x - VISCOSITY * u_xx
        interior_loss = jnp.mean(jnp.square(residual))

        initial_pred = batched_predict(model, batch["initial_coords"])
        initial_loss = jnp.mean(jnp.square(initial_pred - jnp.squeeze(batch["initial_targets"], axis=-1)))

        boundary_pred = batched_predict(model, batch["boundary_coords"])
        boundary_loss = jnp.mean(jnp.square(boundary_pred - jnp.squeeze(batch["boundary_targets"], axis=-1)))

        total = (
            INTERIOR_WEIGHT * interior_loss
            + INITIAL_WEIGHT * initial_loss
            + BOUNDARY_WEIGHT * boundary_loss
        )
        return {
            "total": total,
            "interior": interior_loss,
            "initial": initial_loss,
            "boundary": boundary_loss,
        }

    def loss_and_metrics(model: eqx.Module, step_index: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        metrics = compute_losses(model, sample_batch(step_index))
        return metrics["total"], metrics

    @eqx.filter_jit
    def step_fn(
        model: eqx.Module,
        opt_state: optax.OptState,
        step_index: jax.Array,
        learning_rate: jax.Array,
    ) -> tuple[eqx.Module, optax.OptState, dict[str, jax.Array]]:
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_metrics, has_aux=True)(model, step_index)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, scale_updates(updates, learning_rate))
        metrics = {
            "total": loss,
            "interior": metrics["interior"],
            "initial": metrics["initial"],
            "boundary": metrics["boundary"],
        }
        return model, opt_state, metrics

    return step_fn


def build_phase_runtimes(phases: tuple[OptimizerPhase, ...], pools: SamplingPools) -> list[PhaseRuntime]:
    runtimes = []
    for phase in phases:
        optimizer = build_optimizer(phase)
        step_fn = make_train_step(pools, optimizer)
        runtimes.append(PhaseRuntime(phase=phase, optimizer=optimizer, step_fn=step_fn))
    return runtimes


def warmup_phase_runtimes(runtimes: list[PhaseRuntime], model: eqx.Module) -> None:
    step_index = jnp.asarray(0, dtype=jnp.int32)
    for runtime in runtimes:
        opt_state = runtime.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        _, _, metrics = runtime.step_fn(
            model,
            opt_state,
            step_index,
            jnp.asarray(runtime.phase.lr, dtype=jnp.float32),
        )
        jax.block_until_ready(metrics["total"])


def build_predict_batch() -> Callable[[eqx.Module, jax.Array], jax.Array]:
    @eqx.filter_jit
    def predict_batch(model: eqx.Module, coords: jax.Array) -> jax.Array:
        return jax.vmap(model)(coords)[:, None]

    return predict_batch


def get_peak_memory_mb() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return 0.0
    if not stats:
        return 0.0
    for key in ("peak_bytes_in_use", "bytes_in_use", "peak_unified_bytes_in_use"):
        if key in stats and stats[key] is not None:
            return float(stats[key]) / 1024.0 / 1024.0
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_reference_artifacts(require_existing=True)

    model_config = ModelConfig(
        family=NETWORK_FAMILY,
        input_encoding=INPUT_ENCODING,
        activation=ACTIVATION,
        depth=DEPTH,
        hidden_dim=HIDDEN_DIM,
        fourier_features=FOURIER_FEATURES,
        fourier_scale=FOURIER_SCALE,
        fourier_seed=FOURIER_SEED,
        siren_first_w0=SIREN_FIRST_W0,
        siren_hidden_w0=SIREN_HIDDEN_W0,
    )

    phase_weights = validate_phases(OPTIMIZER_PHASES)
    phase_end_times = []
    elapsed = 0.0
    for weight in phase_weights:
        elapsed += TIME_BUDGET * weight
        phase_end_times.append(elapsed)

    rng = jax.random.PRNGKey(SEED)
    model = build_model(model_config, key=rng)
    pools = build_sampling_pools()
    runtimes = build_phase_runtimes(OPTIMIZER_PHASES, pools)
    warmup_phase_runtimes(runtimes, model)
    predict_batch = build_predict_batch()

    num_params = count_parameters(model)
    device = jax.devices()[0]
    print(f"Backend: {jax.default_backend()}")
    print(f"Device: {device}")
    print(f"Model config: {asdict(model_config)}")
    print(f"Optimizer phases: {[asdict(phase) for phase in OPTIMIZER_PHASES]}")
    print(f"Parameter count: {num_params:,}")
    print(f"Time budget: {TIME_BUDGET}s")

    phase_index = 0
    phase_start_time = 0.0
    runtime = runtimes[phase_index]
    opt_state = runtime.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    training_seconds = 0.0
    step = 0
    smooth_total = 0.0
    ema_beta = 0.9
    last_log = {"total": float("nan"), "interior": float("nan"), "initial": float("nan"), "boundary": float("nan")}

    gc.collect()
    t_start = time.time()

    while training_seconds < TIME_BUDGET:
        if phase_index + 1 < len(runtimes) and training_seconds >= phase_end_times[phase_index]:
            phase_index += 1
            runtime = runtimes[phase_index]
            phase_start_time = phase_end_times[phase_index - 1]
            opt_state = runtime.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        phase_end = phase_end_times[phase_index]
        phase_duration = max(phase_end - phase_start_time, 1e-8)
        phase_progress = min(max((training_seconds - phase_start_time) / phase_duration, 0.0), 1.0)
        learning_rate = runtime.phase.lr * phase_schedule_multiplier(phase_progress)

        step_index = jnp.asarray(step, dtype=jnp.int32)
        learning_rate_array = jnp.asarray(learning_rate, dtype=jnp.float32)

        t0 = time.time()
        model, opt_state, metrics = runtime.step_fn(model, opt_state, step_index, learning_rate_array)
        jax.block_until_ready(metrics["total"])
        dt = time.time() - t0
        training_seconds += dt

        last_log = {name: float(np.asarray(metrics[name])) for name in metrics}
        if not math.isfinite(last_log["total"]):
            print("FAIL")
            raise RuntimeError("Training diverged: non-finite loss encountered.")

        smooth_total = ema_beta * smooth_total + (1.0 - ema_beta) * last_log["total"]
        smooth_total_debiased = smooth_total / (1.0 - ema_beta ** (step + 1))
        pct_done = 100.0 * min(training_seconds / TIME_BUDGET, 1.0)
        remaining = max(0.0, TIME_BUDGET - training_seconds)
        print(
            f"\rstep {step:05d} ({pct_done:5.1f}%) | "
            f"loss: {smooth_total_debiased:.6e} | "
            f"pde: {last_log['interior']:.3e} | "
            f"ic: {last_log['initial']:.3e} | "
            f"bc: {last_log['boundary']:.3e} | "
            f"lr: {learning_rate:.2e} | "
            f"phase: {runtime.phase.name} | "
            f"remaining: {remaining:5.1f}s",
            end="",
            flush=True,
        )

        step += 1
        if step % 200 == 0:
            gc.collect()

    print()

    eval_metrics = evaluate_model(predict_batch, model, batch_size=EVAL_BATCH_SIZE)
    t_end = time.time()
    peak_vram_mb = get_peak_memory_mb()
    optimizer_name = "+".join(phase.kind for phase in OPTIMIZER_PHASES)

    print("---")
    print(f"val_rel_l2:       {eval_metrics['rel_l2']:.6e}")
    print(f"val_rmse:         {eval_metrics['rmse']:.6e}")
    print(f"val_max_abs:      {eval_metrics['max_abs']:.6e}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"network_family:   {NETWORK_FAMILY}")
    print(f"input_encoding:   {INPUT_ENCODING}")
    print(f"optimizer_name:   {optimizer_name}")


if __name__ == "__main__":
    main()
