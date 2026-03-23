# autoresearch-pinns

This repo repurposes [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) for scientific machine learning. The task is operator learning for the 1D viscous Burgers equation: given a viscosity `nu` and an initial condition sampled from a hierarchical Gaussian random field prior, predict the full spatio-temporal solution field `u(x, t)`.

`main` now reflects the large operator-learning setup that was used for the finished 20-experiment cluster run. A fresh clone can rebuild the fixed dataset, rerun the promoted model on that benchmark, and regenerate the committed figures without depending on the old temp workspace.

## Repo Structure

The repo stays intentionally small:

- [`prepare.py`](prepare.py): fixed benchmark harness. It samples the hierarchical prior, solves Burgers with a deterministic finite-volume SSP-RK3 solver, caches the dataset, and owns the evaluation metric.
- [`train.py`](train.py): the editable experiment surface. It contains the surrogate architecture, optimizer schedule, batching, and checkpoint-writing logic.
- [`program.md`](program.md): the autoresearch protocol for long-running search.

## Fixed Benchmark On Main

The tracked benchmark on `main` is the large operator split:

- domain: `x in [-1, 1]`, `t in [0, 1]`
- PDE: `u_t + u u_x - nu u_xx = 0`
- viscosity prior: log-uniform on `[2.5e-3, 5.0e-2]`
- initial-condition prior: hierarchical Gaussian random field with sampled amplitude, lengthscale, and bias
- dataset split: `100000` train, `20000` validation, `500` test
- grids: `128` initial-condition points and a `65 x 128` solution field
- experiment budget: `900` seconds of training per run
- cache path: `~/.cache/autoresearch-burgers/surrogate-large-100k20k500/`

`prepare.py` is fixed during experiments. `train.py` is the only file the search loop should mutate.

## Current Promoted Model

The current tracked `train.py` is the promoted post-search operator configuration from the large run, not the historical search baseline. It is an Equinox/JAX DeepONet-style surrogate with:

- `resmlp` branch network
- `mlp` trunk network
- Fourier coordinate encoding
- branch/trunk hidden width `640`
- latent dimension `448`
- pointwise field supervision plus an auxiliary `t=0` consistency loss

That means a fresh rerun from `main` should target the promoted large-run configuration. The historical search trajectory, including the earlier weaker baseline, is preserved in the committed results bundle.

The keep/discard metric remains **`val_rel_l2`** on the fixed validation split. Test metrics are reported for context only and should not drive the search.

## Rerun From Scratch

Requirements:

- Python `3.14.0` preferred via [`.python-version`](.python-version)
- [`uv`](https://docs.astral.sh/uv/)
- enough CPU/GPU memory for JAX training

Local rerun:

```bash
uv sync
uv run prepare.py --jobs 8
uv run train.py
```

For the full large benchmark, the one-time dataset build is much heavier than a single training run. Local execution works, but the intended path is cluster-backed CPU preparation plus GPU training.

`uv` keeps dependencies in the repo-local `.venv/` instead of the global environment.

## Exact Cluster Replay

For Gautschi, these are the exact replay commands the repo now documents:

```bash
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py doctor --profile gautschi-cpu
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py doctor --profile gautschi-gpu

python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py run-workload \
  --profile gautschi-cpu \
  --workload "build large Burgers surrogate dataset" \
  --prefix burgers-surrogate-large-prepare \
  --command "python3 prepare.py --jobs 32" \
  --submit-arg=--cpus-per-task=32 \
  --submit-arg=--time=08:00:00 \
  --submit-arg=--mem=64G \
  --wait --fetch-logs --tail 200

python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py run-workload \
  --profile gautschi-gpu \
  --workload "train large Burgers surrogate baseline on GPU" \
  --prefix burgers-surrogate-large-baseline \
  --command "python3 prepare.py --help > /dev/null" \
  --command "python3 train.py" \
  --submit-arg=--time=01:00:00 \
  --submit-arg=--mem=48G \
  --wait --fetch-logs --tail 200
```

The lightweight `python3 prepare.py --help > /dev/null` command in the GPU workload is intentional. The cluster runner auto-uploads Python scripts referenced in workload commands, which guarantees that `prepare.py` is staged next to `train.py` for the remote `import prepare`.

After a completed cluster run, download the saved checkpoint bundle back into the repo:

```bash
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py download \
  --run-id <RUN_ID> \
  --remote-path results/checkpoints/<checkpoint-subdir> \
  --local-path results/checkpoints
```

The checkpoint bundle includes machine-readable metadata and saved validation/test predictions so later figure generation does not require rerunning old jobs.

## Committed Large-Run Results

The committed artifact bundle captures the finished 20-experiment cluster search that promoted the current tracked configuration:

- ledger copy: [`artifacts/2026-03-23-large-run/results.tsv`](artifacts/2026-03-23-large-run/results.tsv)
- machine-readable summary: [`artifacts/2026-03-23-large-run/summary.json`](artifacts/2026-03-23-large-run/summary.json)
- progress figure: [`artifacts/2026-03-23-large-run/figure_progress.svg`](artifacts/2026-03-23-large-run/figure_progress.svg)
- baseline vs best field examples: [`artifacts/2026-03-23-large-run/figure_examples_fields.svg`](artifacts/2026-03-23-large-run/figure_examples_fields.svg)
- baseline vs best slice comparisons: [`artifacts/2026-03-23-large-run/figure_examples_slices.svg`](artifacts/2026-03-23-large-run/figure_examples_slices.svg)
- TikZ loop diagram source and PDF:
  - [`artifacts/2026-03-23-large-run/autoresearch_loop_tikz.tex`](artifacts/2026-03-23-large-run/autoresearch_loop_tikz.tex)
  - [`artifacts/2026-03-23-large-run/autoresearch_loop_tikz.pdf`](artifacts/2026-03-23-large-run/autoresearch_loop_tikz.pdf)

Headline numbers from that run:

- historical search baseline: `val_rel_l2 = 2.963700e-02`
- best kept run: `val_rel_l2 = 1.683636e-02`
- relative improvement: `1.76x`
- validation error reduction: `43.19%`

The best run in that search was experiment `17`, commit `a37ff71`, described in the ledger as `widen branch and trunk hidden layers to 640`.

## Regenerate The Figures

The figure generator is committed as [`scripts/make_large_run_figures.py`](scripts/make_large_run_figures.py).

It expects a fetched large-run workspace with the cluster artifacts mirrored locally. Example:

```bash
python3 scripts/make_large_run_figures.py \
  --workspace /tmp/autoresearch-codex-large.woja6k \
  --output-dir artifacts/2026-03-23-large-run
```

This script reads the fetched checkpoint bundles, cross-checks every `results.tsv` row against the saved checkpoint metadata, and regenerates the committed figures.

## License

MIT
