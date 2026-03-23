"""
Autoresearch training script for the Burgers surrogate benchmark.

This baseline learns an operator that maps:
- viscosity nu
- an initial condition sampled from the fixed hierarchical GP prior

to the full Burgers spatio-temporal field on the cached evaluation grid.

Usage:
    uv run train.py
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import prepare as prepare_module
from prepare import (
    FIELD_T_POINTS,
    FIELD_X_POINTS,
    IC_GRID_POINTS,
    TIME_BUDGET,
    ensure_dataset_artifacts,
    evaluate_model,
    field_t_grid,
    load_dataset_artifacts,
    query_coords_grid,
)

jax.config.update("jax_default_matmul_precision", "high")


# ---------------------------------------------------------------------------
# Fixed run-artifact contract
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ROOT = Path("results") / "checkpoints"


# ---------------------------------------------------------------------------
# Editable experiment surface
# ---------------------------------------------------------------------------

SEED = 1337

# Surrogate architecture
MODEL_FAMILY = "deeponet"     # deeponet
BRANCH_FAMILY = "resmlp"      # mlp | resmlp
TRUNK_FAMILY = "mlp"          # mlp | resmlp
COORD_ENCODING = "fourier"    # raw | fourier
ACTIVATION = "silu"           # tanh | silu | gelu | relu
BRANCH_DEPTH = 4
TRUNK_DEPTH = 4
BRANCH_HIDDEN_DIM = 640
TRUNK_HIDDEN_DIM = 640
LATENT_DIM = 448
FOURIER_FEATURES = 96
FOURIER_SCALE = 3.0
FOURIER_SEED = 0

# Data supervision
EXAMPLE_BATCH_SIZE = 24
POINT_BATCH_SIZE = 2048
INITIAL_SLICE_POINTS = 64
EXAMPLE_POOL_MULTIPLIER = 128
POINT_POOL_MULTIPLIER = 64
INITIAL_POOL_MULTIPLIER = 64
LOSS_NAME = "huber"           # mse | huber
HUBER_DELTA = 0.05
INITIAL_SLICE_WEIGHT = 0.25

# Optimization
GRAD_CLIP_NORM = 1.0
EVAL_BATCH_SIZE = 64
SCHEDULER_NAME = "cosine"     # constant | cosine
WARMUP_FRACTION = 0.05
MIN_LR_RATIO = 0.1


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
        name="adamw-main",
        kind="adamw",
        duration_ratio=0.75,
        lr=3e-4,
        weight_decay=1e-5,
    ),
    OptimizerPhase(
        name="adamw-finetune",
        kind="adamw",
        duration_ratio=0.25,
        lr=1e-4,
        weight_decay=0.0,
    ),
)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_family: str
    branch_family: str
    trunk_family: str
    coord_encoding: str
    activation: str
    branch_depth: int
    trunk_depth: int
    branch_hidden_dim: int
    trunk_hidden_dim: int
    latent_dim: int
    fourier_features: int
    fourier_scale: float
    fourier_seed: int


class Normalization(eqx.Module):
    ic_mean: jax.Array
    ic_std: jax.Array
    field_mean: jax.Array
    field_std: jax.Array
    log_viscosity_mean: jax.Array
    log_viscosity_std: jax.Array


class CoordinateEncoder(eqx.Module):
    encoding: str = eqx.field(static=True)
    encoder_matrix: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def __init__(self, encoding: str, *, fourier_features: int, fourier_scale: float, fourier_seed: int):
        self.encoding = encoding
        if encoding == "fourier":
            rng = np.random.default_rng(fourier_seed)
            matrix = (rng.standard_normal((2, fourier_features)) * fourier_scale).astype(np.float32)
            self.encoder_matrix = tuple(tuple(float(value) for value in row) for row in matrix)
        else:
            self.encoder_matrix = None

    def __call__(self, coords: jax.Array) -> jax.Array:
        coords = coords.astype(jnp.float32)
        if self.encoding == "raw":
            return coords
        if self.encoding != "fourier":
            raise ValueError(f"Unsupported coordinate encoding: {self.encoding}")
        matrix = jnp.asarray(self.encoder_matrix, dtype=jnp.float32)
        projected = 2.0 * math.pi * (coords @ matrix)
        return jnp.concatenate((coords, jnp.sin(projected), jnp.cos(projected)), axis=-1)


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


class PlainMLP(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_name: str,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=make_activation(activation_name),
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


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
        return activation(x + residual)


class ResidualMLP(eqx.Module):
    stem: eqx.nn.Linear
    blocks: tuple[ResidualBlock, ...]
    head: eqx.nn.Linear
    activation_name: str = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_name: str,
        key: jax.Array,
    ):
        if depth < 1:
            raise ValueError("ResidualMLP depth must be at least 1.")
        keys = jax.random.split(key, depth + 2)
        self.stem = eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])
        self.blocks = tuple(
            ResidualBlock(hidden_dim, activation_name, key=block_key)
            for block_key in keys[1:-1]
        )
        self.head = eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])
        self.activation_name = activation_name

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = make_activation(self.activation_name)
        x = activation(self.stem(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_feature_network(
    family: str,
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    depth: int,
    activation_name: str,
    key: jax.Array,
) -> eqx.Module:
    family = family.lower()
    if family == "mlp":
        return PlainMLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation_name=activation_name,
            key=key,
        )
    if family == "resmlp":
        return ResidualMLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation_name=activation_name,
            key=key,
        )
    raise ValueError(f"Unsupported network family: {family}")


class DeepONetSurrogate(eqx.Module):
    normalization: Normalization
    coord_encoder: CoordinateEncoder
    branch_net: eqx.Module
    trunk_net: eqx.Module
    latent_dim: int = eqx.field(static=True)
    field_t_points: int = eqx.field(static=True)
    field_x_points: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, normalization: Normalization, *, key: jax.Array):
        branch_key, trunk_key = jax.random.split(key)
        coord_input_dim = 2 if config.coord_encoding == "raw" else 2 + 2 * config.fourier_features

        self.normalization = normalization
        self.coord_encoder = CoordinateEncoder(
            config.coord_encoding,
            fourier_features=config.fourier_features,
            fourier_scale=config.fourier_scale,
            fourier_seed=config.fourier_seed,
        )
        self.branch_net = build_feature_network(
            config.branch_family,
            IC_GRID_POINTS + 1,
            config.latent_dim + 1,
            hidden_dim=config.branch_hidden_dim,
            depth=config.branch_depth,
            activation_name=config.activation,
            key=branch_key,
        )
        self.trunk_net = build_feature_network(
            config.trunk_family,
            coord_input_dim,
            config.latent_dim,
            hidden_dim=config.trunk_hidden_dim,
            depth=config.trunk_depth,
            activation_name=config.activation,
            key=trunk_key,
        )
        self.latent_dim = config.latent_dim
        self.field_t_points = FIELD_T_POINTS
        self.field_x_points = FIELD_X_POINTS

    def normalize_inputs(self, viscosity: jax.Array, initial_conditions: jax.Array) -> tuple[jax.Array, jax.Array]:
        log_viscosity = jnp.log(jnp.maximum(viscosity, 1e-12))
        normalized_viscosity = (
            log_viscosity - self.normalization.log_viscosity_mean
        ) / self.normalization.log_viscosity_std
        normalized_initial = (
            initial_conditions - self.normalization.ic_mean[None, :]
        ) / self.normalization.ic_std[None, :]
        return normalized_viscosity[:, None], normalized_initial

    def normalize_field(self, values: jax.Array) -> jax.Array:
        return (values - self.normalization.field_mean) / self.normalization.field_std

    def denormalize_field(self, values: jax.Array) -> jax.Array:
        return values * self.normalization.field_std + self.normalization.field_mean

    def branch_coefficients(self, viscosity: jax.Array, initial_conditions: jax.Array) -> jax.Array:
        viscosity_feature, normalized_initial = self.normalize_inputs(viscosity, initial_conditions)
        branch_inputs = jnp.concatenate((normalized_initial, viscosity_feature), axis=-1)
        return jax.vmap(self.branch_net)(branch_inputs)

    def trunk_features_encoded(self, encoded_coords: jax.Array) -> jax.Array:
        return jax.vmap(self.trunk_net)(encoded_coords.astype(jnp.float32))

    def trunk_features(self, coords: jax.Array) -> jax.Array:
        return self.trunk_features_encoded(self.coord_encoder(coords))

    def combine_branch_trunk_normalized(self, branch: jax.Array, trunk: jax.Array) -> jax.Array:
        basis_coeffs = branch[:, :-1]
        bias = branch[:, -1:]
        return jnp.einsum("bl,ql->bq", basis_coeffs, trunk) / math.sqrt(self.latent_dim) + bias

    def predict_points_normalized(
        self,
        viscosity: jax.Array,
        initial_conditions: jax.Array,
        coords: jax.Array,
    ) -> jax.Array:
        return self.predict_points_normalized_encoded(
            viscosity,
            initial_conditions,
            self.coord_encoder(coords),
        )

    def predict_points_normalized_encoded(
        self,
        viscosity: jax.Array,
        initial_conditions: jax.Array,
        encoded_coords: jax.Array,
    ) -> jax.Array:
        branch = self.branch_coefficients(viscosity, initial_conditions)
        trunk = self.trunk_features_encoded(encoded_coords)
        return self.combine_branch_trunk_normalized(branch, trunk)

    def predict_fields(
        self,
        viscosity: jax.Array,
        initial_conditions: jax.Array,
        coords: jax.Array,
    ) -> jax.Array:
        return self.predict_fields_encoded(
            viscosity,
            initial_conditions,
            self.coord_encoder(coords),
        )

    def predict_fields_encoded(
        self,
        viscosity: jax.Array,
        initial_conditions: jax.Array,
        encoded_coords: jax.Array,
    ) -> jax.Array:
        pred_norm = self.predict_points_normalized_encoded(viscosity, initial_conditions, encoded_coords)
        pred = self.denormalize_field(pred_norm)
        return pred.reshape((pred.shape[0], self.field_t_points, self.field_x_points))


def build_model(config: ModelConfig, normalization: Normalization, *, key: jax.Array) -> eqx.Module:
    if config.model_family != "deeponet":
        raise ValueError(f"Unsupported model family: {config.model_family}")
    return DeepONetSurrogate(config, normalization, key=key)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class TrainingData(eqx.Module):
    train_viscosity: jax.Array
    train_initial_conditions: jax.Array
    train_fields_normalized: jax.Array
    encoded_query_coords: jax.Array
    encoded_initial_coords: jax.Array
    train_size: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    initial_size: int = eqx.field(static=True)


class SamplingPools(eqx.Module):
    example_pool: jax.Array
    example_pool_size: int = eqx.field(static=True)
    point_pool: jax.Array
    point_pool_size: int = eqx.field(static=True)
    initial_pool: jax.Array
    initial_pool_size: int = eqx.field(static=True)


class TrainInputs(eqx.Module):
    data: TrainingData
    pools: SamplingPools


@dataclass(frozen=True)
class PhaseRuntime:
    phase: OptimizerPhase
    optimizer: optax.GradientTransformation
    step_fn: Callable[[eqx.Module, optax.OptState, jax.Array, jax.Array], tuple[eqx.Module, optax.OptState, dict[str, jax.Array]]]


def build_normalization(dataset: dict[str, np.ndarray]) -> Normalization:
    return Normalization(
        ic_mean=jnp.asarray(dataset["ic_mean"], dtype=jnp.float32),
        ic_std=jnp.asarray(dataset["ic_std"], dtype=jnp.float32),
        field_mean=jnp.asarray(dataset["field_mean"], dtype=jnp.float32),
        field_std=jnp.asarray(dataset["field_std"], dtype=jnp.float32),
        log_viscosity_mean=jnp.asarray(dataset["log_viscosity_mean"], dtype=jnp.float32),
        log_viscosity_std=jnp.asarray(dataset["log_viscosity_std"], dtype=jnp.float32),
    )


def build_training_data(
    dataset: dict[str, np.ndarray],
    normalization: Normalization,
    model_config: ModelConfig,
) -> TrainingData:
    train_fields = jnp.asarray(dataset["train_fields"], dtype=jnp.float32)
    train_fields_normalized = (
        train_fields - normalization.field_mean
    ) / normalization.field_std
    query_coords = jnp.asarray(dataset["query_coords"], dtype=jnp.float32)
    encoder = CoordinateEncoder(
        model_config.coord_encoding,
        fourier_features=model_config.fourier_features,
        fourier_scale=model_config.fourier_scale,
        fourier_seed=model_config.fourier_seed,
    )
    encoded_query_coords = encoder(query_coords)
    encoded_initial_coords = encoded_query_coords[:FIELD_X_POINTS]
    return TrainingData(
        train_viscosity=jnp.asarray(dataset["train_viscosity"], dtype=jnp.float32),
        train_initial_conditions=jnp.asarray(dataset["train_initial_conditions"], dtype=jnp.float32),
        train_fields_normalized=train_fields_normalized.reshape((train_fields.shape[0], -1)),
        encoded_query_coords=encoded_query_coords,
        encoded_initial_coords=encoded_initial_coords,
        train_size=int(dataset["train_viscosity"].shape[0]),
        query_size=int(encoded_query_coords.shape[0]),
        initial_size=FIELD_X_POINTS,
    )


def make_index_pool(size: int, *, batch_size: int, multiplier: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    blocks = []
    total = 0
    required = max(size, batch_size * multiplier)
    while total < required:
        blocks.append(rng.permutation(size).astype(np.int32))
        total += size
    return jnp.asarray(np.concatenate(blocks, axis=0))


def build_sampling_pools(data: TrainingData) -> SamplingPools:
    return SamplingPools(
        example_pool=make_index_pool(
            data.train_size,
            batch_size=EXAMPLE_BATCH_SIZE,
            multiplier=EXAMPLE_POOL_MULTIPLIER,
            seed=SEED + 101,
        ),
        example_pool_size=max(data.train_size, EXAMPLE_BATCH_SIZE * EXAMPLE_POOL_MULTIPLIER),
        point_pool=make_index_pool(
            data.query_size,
            batch_size=POINT_BATCH_SIZE,
            multiplier=POINT_POOL_MULTIPLIER,
            seed=SEED + 202,
        ),
        point_pool_size=max(data.query_size, POINT_BATCH_SIZE * POINT_POOL_MULTIPLIER),
        initial_pool=make_index_pool(
            data.initial_size,
            batch_size=INITIAL_SLICE_POINTS,
            multiplier=INITIAL_POOL_MULTIPLIER,
            seed=SEED + 303,
        ),
        initial_pool_size=max(data.initial_size, INITIAL_SLICE_POINTS * INITIAL_POOL_MULTIPLIER),
    )


def take_cyclic(pool: jax.Array, batch_size: int, start: jax.Array) -> jax.Array:
    indices = (start + jnp.arange(batch_size, dtype=jnp.int32)) % pool.shape[0]
    return jnp.take(pool, indices, axis=0)


def count_parameters(model: eqx.Module) -> int:
    arrays = eqx.filter(model, eqx.is_inexact_array)
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(arrays)))


def loss_values(pred: jax.Array, target: jax.Array) -> jax.Array:
    if LOSS_NAME == "mse":
        return jnp.square(pred - target)
    if LOSS_NAME == "huber":
        return optax.huber_loss(pred, target, delta=HUBER_DELTA)
    raise ValueError(f"Unsupported loss: {LOSS_NAME}")


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


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


def scale_updates(updates: Any, learning_rate: jax.Array) -> Any:
    return jax.tree_util.tree_map(
        lambda update: None if update is None else learning_rate * update,
        updates,
    )


def sample_batch(train_inputs: TrainInputs, step_index: jax.Array) -> dict[str, jax.Array]:
    data = train_inputs.data
    pools = train_inputs.pools

    example_start = step_index * EXAMPLE_BATCH_SIZE
    point_start = step_index * (POINT_BATCH_SIZE * 3)
    initial_start = step_index * (INITIAL_SLICE_POINTS * 5)

    example_indices = take_cyclic(pools.example_pool, EXAMPLE_BATCH_SIZE, example_start)
    point_indices = take_cyclic(pools.point_pool, POINT_BATCH_SIZE, point_start)
    initial_indices = take_cyclic(pools.initial_pool, INITIAL_SLICE_POINTS, initial_start)

    batch_viscosity = jnp.take(data.train_viscosity, example_indices, axis=0)
    batch_initial = jnp.take(data.train_initial_conditions, example_indices, axis=0)
    batch_fields = jnp.take(data.train_fields_normalized, example_indices, axis=0)
    batch_targets = jnp.take(batch_fields, point_indices, axis=1)
    initial_targets = jnp.take(batch_initial, initial_indices, axis=1)
    return {
        "viscosity": batch_viscosity,
        "initial_conditions": batch_initial,
        "encoded_coords": jnp.take(data.encoded_query_coords, point_indices, axis=0),
        "targets": batch_targets,
        "encoded_initial_coords": jnp.take(data.encoded_initial_coords, initial_indices, axis=0),
        "initial_targets": initial_targets,
    }


def compute_losses(model: DeepONetSurrogate, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    branch = model.branch_coefficients(batch["viscosity"], batch["initial_conditions"])

    pred = model.combine_branch_trunk_normalized(branch, model.trunk_features_encoded(batch["encoded_coords"]))
    supervised_loss = jnp.mean(loss_values(pred, batch["targets"]))

    initial_pred = model.combine_branch_trunk_normalized(
        branch,
        model.trunk_features_encoded(batch["encoded_initial_coords"]),
    )
    initial_target_norm = model.normalize_field(batch["initial_targets"])
    initial_loss = jnp.mean(loss_values(initial_pred, initial_target_norm))

    total = supervised_loss + INITIAL_SLICE_WEIGHT * initial_loss
    return {
        "total": total,
        "supervised": supervised_loss,
        "initial_slice": initial_loss,
    }


def make_train_step(
    optimizer: optax.GradientTransformation,
) -> Callable[[TrainInputs, eqx.Module, optax.OptState, jax.Array, jax.Array], tuple[eqx.Module, optax.OptState, dict[str, jax.Array]]]:
    def loss_and_metrics(model: DeepONetSurrogate, batch: dict[str, jax.Array]) -> tuple[jax.Array, dict[str, jax.Array]]:
        metrics = compute_losses(model, batch)
        return metrics["total"], metrics

    # Keep the large training arrays out of the lowered program constants.
    @eqx.filter_jit
    def step_fn(
        train_inputs: TrainInputs,
        model: DeepONetSurrogate,
        opt_state: optax.OptState,
        step_index: jax.Array,
        learning_rate: jax.Array,
    ) -> tuple[DeepONetSurrogate, optax.OptState, dict[str, jax.Array]]:
        batch = sample_batch(train_inputs, step_index)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_metrics, has_aux=True)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, scale_updates(updates, learning_rate))
        return model, opt_state, {
            "total": loss,
            "supervised": metrics["supervised"],
            "initial_slice": metrics["initial_slice"],
        }

    return step_fn


def build_phase_runtimes(
    phases: tuple[OptimizerPhase, ...],
) -> list[PhaseRuntime]:
    runtimes = []
    for phase in phases:
        optimizer = build_optimizer(phase)
        step_fn = make_train_step(optimizer)
        runtimes.append(PhaseRuntime(phase=phase, optimizer=optimizer, step_fn=step_fn))
    return runtimes


def warmup_phase_runtimes(runtimes: list[PhaseRuntime], train_inputs: TrainInputs, model: DeepONetSurrogate) -> None:
    step_index = jnp.asarray(0, dtype=jnp.int32)
    for runtime in runtimes:
        opt_state = runtime.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        _, _, metrics = runtime.step_fn(
            train_inputs,
            model,
            opt_state,
            step_index,
            jnp.asarray(runtime.phase.lr, dtype=jnp.float32),
        )
        jax.block_until_ready(metrics["total"])


def build_predict_fields(encoded_query_coords: jax.Array) -> Callable[[DeepONetSurrogate, jax.Array, jax.Array], jax.Array]:
    @eqx.filter_jit
    def predict_fields(model: DeepONetSurrogate, viscosity: jax.Array, initial_conditions: jax.Array) -> jax.Array:
        return model.predict_fields_encoded(viscosity, initial_conditions, encoded_query_coords)

    return predict_fields


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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def get_git_commit_short() -> str:
    commit = _git_output("rev-parse", "--short", "HEAD")
    return commit if commit else "nogit"


def get_git_dirty() -> bool:
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    return bool(status)


def predict_dataset_split(
    predict_fields: Callable[[DeepONetSurrogate, jax.Array, jax.Array], jax.Array],
    model: DeepONetSurrogate,
    dataset: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    viscosity = dataset[f"{split}_viscosity"]
    initial_conditions = dataset[f"{split}_initial_conditions"]
    targets = dataset[f"{split}_fields"]

    preds = []
    for start in range(0, viscosity.shape[0], batch_size):
        stop = min(start + batch_size, viscosity.shape[0])
        batch_viscosity = jnp.asarray(viscosity[start:stop], dtype=jnp.float32)
        batch_initial = jnp.asarray(initial_conditions[start:stop], dtype=jnp.float32)
        batch_pred = predict_fields(model, batch_viscosity, batch_initial)
        preds.append(np.asarray(jax.device_get(batch_pred), dtype=np.float32))

    return {
        "viscosity": viscosity.astype(np.float32, copy=False),
        "initial_conditions": initial_conditions.astype(np.float32, copy=False),
        "targets": targets.astype(np.float32, copy=False),
        "predictions": np.concatenate(preds, axis=0).astype(np.float32, copy=False),
    }


def make_checkpoint_dir(run_started_at: float) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(run_started_at))
    run_id = f"{timestamp}-{get_git_commit_short()}-pid{os.getpid()}"
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir


def save_run_checkpoint(
    *,
    model: DeepONetSurrogate,
    opt_state: optax.OptState,
    predict_fields: Callable[[DeepONetSurrogate, jax.Array, jax.Array], jax.Array],
    model_config: ModelConfig,
    device: jax.Device,
    phase_index: int,
    step: int,
    num_params: int,
    training_seconds: float,
    total_seconds: float,
    peak_vram_mb: float,
    optimizer_name: str,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    last_log: dict[str, float],
    run_started_at: float,
    run_finished_at: float,
) -> dict[str, str]:
    checkpoint_dir = make_checkpoint_dir(run_started_at)
    model_file = checkpoint_dir / "model.eqx"
    opt_state_file = checkpoint_dir / "opt_state.eqx"
    predictions_file = checkpoint_dir / "predictions.npz"
    metadata_file = checkpoint_dir / "metadata.json"
    train_snapshot_file = checkpoint_dir / "train_snapshot.py"
    prepare_snapshot_file = checkpoint_dir / "prepare_snapshot.py"

    dataset = load_dataset_artifacts()
    val_payload = predict_dataset_split(predict_fields, model, dataset, "val", batch_size=EVAL_BATCH_SIZE)
    test_payload = predict_dataset_split(predict_fields, model, dataset, "test", batch_size=EVAL_BATCH_SIZE)

    eqx.tree_serialise_leaves(model_file, model)
    eqx.tree_serialise_leaves(opt_state_file, opt_state)
    np.savez(
        predictions_file,
        ic_x=dataset["ic_x"].astype(np.float32, copy=False),
        field_x=dataset["field_x"].astype(np.float32, copy=False),
        field_t=dataset["field_t"].astype(np.float32, copy=False),
        val_viscosity=val_payload["viscosity"],
        val_initial_conditions=val_payload["initial_conditions"],
        val_targets=val_payload["targets"],
        val_predictions=val_payload["predictions"],
        test_viscosity=test_payload["viscosity"],
        test_initial_conditions=test_payload["initial_conditions"],
        test_targets=test_payload["targets"],
        test_predictions=test_payload["predictions"],
    )
    train_snapshot_file.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    prepare_snapshot_file.write_text(
        Path(prepare_module.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "command": "uv run train.py",
        "working_directory": str(Path.cwd().resolve()),
        "git_commit_short": get_git_commit_short(),
        "git_dirty": get_git_dirty(),
        "run_started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(run_started_at)),
        "run_finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(run_finished_at)),
        "seed": SEED,
        "time_budget_seconds": TIME_BUDGET,
        "device": str(device),
        "backend": jax.default_backend(),
        "model_config": asdict(model_config),
        "optimizer_phases": [asdict(phase) for phase in OPTIMIZER_PHASES],
        "active_phase_index": phase_index,
        "active_phase_name": OPTIMIZER_PHASES[phase_index].name,
        "optimizer_name": optimizer_name,
        "num_steps": step,
        "num_params": num_params,
        "num_params_m": num_params / 1e6,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "peak_vram_mb": peak_vram_mb,
        "last_train_metrics": last_log,
        "eval_metrics": {
            "val_rel_l2": val_metrics["rel_l2"],
            "val_rmse": val_metrics["rmse"],
            "val_max_abs": val_metrics["max_abs"],
            "val_worst_rel_l2": val_metrics["worst_rel_l2"],
            "test_rel_l2": test_metrics["rel_l2"],
            "test_rmse": test_metrics["rmse"],
            "test_max_abs": test_metrics["max_abs"],
            "test_worst_rel_l2": test_metrics["worst_rel_l2"],
        },
        "files": {
            "model": str(model_file.resolve()),
            "opt_state": str(opt_state_file.resolve()),
            "predictions": str(predictions_file.resolve()),
            "train_snapshot": str(train_snapshot_file.resolve()),
            "prepare_snapshot": str(prepare_snapshot_file.resolve()),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "model_file": str(model_file.resolve()),
        "opt_state_file": str(opt_state_file.resolve()),
        "predictions_file": str(predictions_file.resolve()),
        "metadata_file": str(metadata_file.resolve()),
    }


def load_checkpoint(checkpoint_dir: str | Path) -> tuple[eqx.Module, optax.OptState, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = load_dataset_artifacts()
    normalization = build_normalization(dataset)
    model_config = ModelConfig(**metadata["model_config"])
    model = build_model(model_config, normalization, key=jax.random.PRNGKey(int(metadata["seed"])))
    model = eqx.tree_deserialise_leaves(checkpoint_dir / "model.eqx", model)

    phase_index = int(metadata["active_phase_index"])
    phases = tuple(OptimizerPhase(**phase) for phase in metadata["optimizer_phases"])
    optimizer = build_optimizer(phases[phase_index])
    opt_state_template = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state = eqx.tree_deserialise_leaves(checkpoint_dir / "opt_state.eqx", opt_state_template)
    return model, opt_state, metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dataset_artifacts(require_existing=True)
    dataset = load_dataset_artifacts()
    normalization = build_normalization(dataset)

    model_config = ModelConfig(
        model_family=MODEL_FAMILY,
        branch_family=BRANCH_FAMILY,
        trunk_family=TRUNK_FAMILY,
        coord_encoding=COORD_ENCODING,
        activation=ACTIVATION,
        branch_depth=BRANCH_DEPTH,
        trunk_depth=TRUNK_DEPTH,
        branch_hidden_dim=BRANCH_HIDDEN_DIM,
        trunk_hidden_dim=TRUNK_HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        fourier_features=FOURIER_FEATURES,
        fourier_scale=FOURIER_SCALE,
        fourier_seed=FOURIER_SEED,
    )
    data = build_training_data(dataset, normalization, model_config)

    phase_weights = validate_phases(OPTIMIZER_PHASES)
    phase_end_times = []
    elapsed = 0.0
    for weight in phase_weights:
        elapsed += TIME_BUDGET * weight
        phase_end_times.append(elapsed)

    rng = jax.random.PRNGKey(SEED)
    model = build_model(model_config, normalization, key=rng)
    pools = build_sampling_pools(data)
    train_inputs = TrainInputs(data=data, pools=pools)
    runtimes = build_phase_runtimes(OPTIMIZER_PHASES)
    warmup_phase_runtimes(runtimes, train_inputs, model)
    predict_fields = build_predict_fields(data.encoded_query_coords)

    num_params = count_parameters(model)
    device = jax.devices()[0]
    print(f"Backend: {jax.default_backend()}")
    print(f"Device: {device}")
    print(f"Model config: {asdict(model_config)}")
    print(f"Optimizer phases: {[asdict(phase) for phase in OPTIMIZER_PHASES]}")
    print(f"Train samples: {data.train_size}")
    print(f"Query points: {data.query_size}")
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
    last_log = {"total": float("nan"), "supervised": float("nan"), "initial_slice": float("nan")}

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
        model, opt_state, metrics = runtime.step_fn(
            train_inputs,
            model,
            opt_state,
            step_index,
            learning_rate_array,
        )
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
            f"field: {last_log['supervised']:.3e} | "
            f"t0: {last_log['initial_slice']:.3e} | "
            f"lr: {learning_rate:.2e} | "
            f"phase: {runtime.phase.name} | "
            f"remaining: {remaining:5.1f}s",
            end="",
            flush=True,
        )

        step += 1
        if step % 100 == 0:
            gc.collect()

    print()

    val_metrics = evaluate_model(predict_fields, model, split="val", batch_size=EVAL_BATCH_SIZE)
    test_metrics = evaluate_model(predict_fields, model, split="test", batch_size=EVAL_BATCH_SIZE)
    t_end = time.time()
    total_seconds = t_end - t_start
    peak_vram_mb = get_peak_memory_mb()
    optimizer_name = "+".join(phase.kind for phase in OPTIMIZER_PHASES)
    checkpoint_files = save_run_checkpoint(
        model=model,
        opt_state=opt_state,
        predict_fields=predict_fields,
        model_config=model_config,
        device=device,
        phase_index=phase_index,
        step=step,
        num_params=num_params,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb=peak_vram_mb,
        optimizer_name=optimizer_name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        last_log=last_log,
        run_started_at=t_start,
        run_finished_at=t_end,
    )

    print("---")
    print(f"val_rel_l2:       {val_metrics['rel_l2']:.6e}")
    print(f"val_rmse:         {val_metrics['rmse']:.6e}")
    print(f"val_max_abs:      {val_metrics['max_abs']:.6e}")
    print(f"val_worst_rel_l2: {val_metrics['worst_rel_l2']:.6e}")
    print(f"test_rel_l2:      {test_metrics['rel_l2']:.6e}")
    print(f"test_rmse:        {test_metrics['rmse']:.6e}")
    print(f"test_max_abs:     {test_metrics['max_abs']:.6e}")
    print(f"test_worst_rel_l2:{test_metrics['worst_rel_l2']:.6e}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"model_family:     {MODEL_FAMILY}")
    print(f"branch_family:    {BRANCH_FAMILY}")
    print(f"trunk_family:     {TRUNK_FAMILY}")
    print(f"coord_encoding:   {COORD_ENCODING}")
    print(f"optimizer_name:   {optimizer_name}")
    print(f"checkpoint_dir:   {checkpoint_files['checkpoint_dir']}")
    print(f"checkpoint_meta:  {checkpoint_files['metadata_file']}")
    print(f"prediction_file:  {checkpoint_files['predictions_file']}")


if __name__ == "__main__":
    main()
