# autoresearch-burgers

This repo repurposes [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) for a scientific machine learning benchmark: using JAX-based physics-informed neural networks to solve the 1D viscous Burgers equation accurately.

The core autoresearch idea stays the same:

- `prepare.py` is the fixed harness.
- `train.py` is the single file the agent edits.
- `program.md` is the human-authored research-org instruction set.

Instead of language-model pretraining, the benchmark is now:

- PDE: `u_t + u u_x - nu u_xx = 0`
- Domain: `x in [-1, 1]`, `t in [0, 1]`
- Viscosity: `nu = 0.01 / pi`
- Initial condition: `u(x, 0) = -sin(pi x)`
- Boundary conditions: `u(-1, t) = u(1, t) = 0`

## How it works

The repo remains deliberately small and centered on three files:

- **`prepare.py`** — fixed problem definition, deterministic sampling helpers, one-time creation of the cached Burgers reference solution, and the fixed validation metric. It is Torch-free and writes NumPy-based cache artifacts. Do not modify during experiments.
- **`train.py`** — the agent-edited experiment surface. This is where you change the network family, input encoding, optimizer phases, collocation counts, loss weights, and training algorithm. The model stack is Equinox + JAX, and optimization uses Optax.
- **`program.md`** — the instructions you hand to the coding agent running the experiment loop.

By design, each training run still gets a **fixed 5-minute wall-clock budget**. That keeps experiments comparable even when the agent changes the network architecture or optimizer strategy.

The primary metric is now **`val_rel_l2`**: relative L2 error on a fixed validation grid built from a cached Burgers reference solution. Lower is better.

## Quick start

Requirements: Python 3.14+, [`uv`](https://docs.astral.sh/uv/), and enough compute to run repeated JAX PINN training loops. `uv` keeps dependencies in the repo-local environment instead of the global Python install. On macOS and Windows the default dependency set uses CPU-capable JAX. On Linux the default dependency set follows the official CUDA 13 JAX plugin path so GPU-backed experiments are ready by default.

```bash
# 1. Install dependencies
uv sync

# 2. Build the fixed Burgers reference artifacts (one-time)
uv run prepare.py

# 3. Run a single baseline PINN experiment (~5 min)
uv run train.py
```

The preparation step writes fixed artifacts to `~/.cache/autoresearch-burgers/reference/`. The local environment will live under `.venv/` once `uv sync` completes.

## Research surface

The point of this setup is to let the agent research PINN design choices without changing the benchmark harness. In `train.py`, the agent can explore:

- network family: `mlp`, `resmlp`, `siren`
- input encoding: raw coordinates or Fourier features
- width, depth, activation, SIREN frequency scaling
- optimizer phases: `adam`, `adamw`, `rmsprop`, `sgd`
- learning-rate schedule and warmup/cooldown behavior
- collocation counts, loss weights, gradient clipping, and sampling method

The fixed harness in `prepare.py` exposes deterministic point samplers and a fixed `evaluate_model` routine, while `train.py` keeps the model and optimizer choices agent-controlled.

## Output contract

At the end of every run, `train.py` prints a summary in the form:

```text
---
val_rel_l2:       1.234567e-02
val_rmse:         8.765432e-03
val_max_abs:      4.321000e-02
training_seconds: 300.0
total_seconds:    304.5
peak_vram_mb:     812.4
num_steps:        1450
num_params_M:     0.123
network_family:   mlp
input_encoding:   raw
optimizer_name:   adam
```

`program.md` and the experiment loop should treat `val_rel_l2` as the only keep/discard metric.

## Cached reference solution

`prepare.py` builds a deterministic Burgers reference solution using a high-resolution finite-volume SSP-RK3 solver and stores:

- `burgers_reference.npz` — validation coordinates and reference values
- `manifest.json` — machine-readable problem and artifact metadata

Training refuses to start if these artifacts are missing or stale. That keeps runs traceable and comparable.

## Design choices

- **Single editable file**: the agent still modifies only `train.py`.
- **Fixed benchmark**: `prepare.py` owns the PDE, domain, boundary conditions, and validation metric.
- **Fixed time budget**: every experiment runs for the same wall-clock budget.
- **Traceable artifacts**: the reference solution is cached with a manifest so future runs can verify exactly what they are evaluating against.

## Running the agent

Start your coding agent in this repo and prompt it to read `program.md`, verify the cached reference artifacts, create a run branch, and begin experimenting on `train.py`.

## License

MIT
