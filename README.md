# autoresearch-pinns

This repo repurposes [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) for a harder scientific machine learning benchmark: learning a surrogate operator for the 1D viscous Burgers equation.

The fixed task is:

- input 1: viscosity `nu`
- input 2: an initial condition `u(x, 0)` sampled from a hierarchical Gaussian random field prior
- output: the full Burgers solution field `u(x, t)` on a fixed spatio-temporal grid

The initial-condition prior is hierarchical:

1. sample GP kernel hyperparameters from a prior
2. conditioned on those hyperparameters, sample an initial condition from a Gaussian process
3. enforce the zero-boundary structure needed by the Burgers benchmark

That makes the problem substantially harder than fitting a single PDE instance with a PINN. The agent now has to learn an operator over a family of PDEs and initial conditions instead of a single solution.

## Repo structure

The repo remains deliberately small:

- **`prepare.py`**: fixed benchmark harness. It samples the hierarchical GP prior, solves Burgers with a deterministic finite-volume SSP-RK3 solver, caches the dataset, and owns the evaluation metric. Do not modify it during experiments.
- **`train.py`**: the only editable experiment surface. It contains the surrogate architecture, optimizer schedule, minibatching strategy, and checkpoint-writing logic.
- **`program.md`**: the experiment protocol for the agent.

## Fixed benchmark

The fixed harness in `prepare.py` defines:

- domain: `x in [-1, 1]`, `t in [0, 1]`
- PDE: `u_t + u u_x - nu u_xx = 0`
- viscosity prior: log-uniform on `[2.5e-3, 5.0e-2]`
- initial-condition prior: hierarchical Gaussian random field with sampled amplitude, lengthscale, and bias
- dataset split: `512` train, `8` validation, `8` test
- grid sizes: `128` initial-condition points, `65 x 128` solution field
- experiment budget: `900` seconds per run

The cached dataset lives under `~/.cache/autoresearch-burgers/surrogate/`.

## Baseline model

The current baseline in `train.py` is an Equinox/JAX DeepONet-style surrogate:

- a branch network consumes the discretized initial condition plus viscosity
- a trunk network consumes space-time coordinates
- their latent interaction reconstructs the full field
- training uses pointwise supervision on sampled field coordinates plus an auxiliary `t=0` consistency loss

This is intentionally just a baseline. The agent is expected to search over branch/trunk family, width, depth, latent size, coordinate encoding, loss, batching, and optimizer schedule inside `train.py`.

## Checkpoint contract

Every completed run writes an untracked checkpoint bundle under `results/checkpoints/` with:

- serialized Equinox model leaves
- serialized optimizer state
- machine-readable metadata
- snapshots of `train.py` and `prepare.py`
- cached validation and test predictions for later inspection and figure generation

This is part of the harness contract. Experiments should stay traceable without rerunning old jobs.

## Local usage

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and enough compute for JAX training.

```bash
# Install the local environment
uv sync

# Build the fixed surrogate dataset
uv run prepare.py --jobs 8

# Run one baseline experiment
uv run train.py
```

`uv` keeps dependencies in the repo-local `.venv/` instead of the global Python environment.

## Cluster usage

This benchmark is intended to run on a SLURM cluster:

- build the cached dataset on CPU
- run surrogate training on GPU

The repo includes the `cluster-slurm` skill so the agent can plan, submit, monitor, and fetch cluster workloads reproducibly.

For the current Gautschi setup, the working high-level commands are:

```bash
# CPU dataset build
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py run-workload \
  --profile gautschi-cpu \
  --workload "build Burgers surrogate dataset" \
  --prefix burgers-surrogate-prepare \
  --command "python3 prepare.py --jobs 16" \
  --submit-arg=--cpus-per-task=16 \
  --submit-arg=--time=01:00:00 \
  --submit-arg=--mem=32G \
  --wait --fetch-logs --tail 200

# GPU baseline training
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py run-workload \
  --profile gautschi-gpu \
  --workload "train Burgers surrogate baseline on GPU" \
  --prefix burgers-surrogate-baseline \
  --command "python3 prepare.py --help > /dev/null" \
  --command "python3 train.py" \
  --submit-arg=--time=01:00:00 \
  --submit-arg=--mem=48G \
  --wait --fetch-logs --tail 200
```

The lightweight `python3 prepare.py --help > /dev/null` command in the GPU run is intentional. The cluster runner auto-uploads Python scripts referenced by workload commands, so this guarantees that `prepare.py` is staged next to `train.py` for the remote `import prepare`.

## Objective

The keep/discard metric is:

- **`val_rel_l2`** on the fixed validation split

Lower is better. Test metrics are reported for context only and should not drive the search loop.

## License

MIT
