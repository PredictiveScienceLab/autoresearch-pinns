# autoresearch-pinns

This repo repurposes [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) for scientific machine learning. The core task here is operator learning for the 1D viscous Burgers equation: given a viscosity `nu` and an initial condition sampled from a hierarchical Gaussian random field prior, predict the full spatio-temporal solution field `u(x, t)`.

The repo is intentionally small:

- [`prepare.py`](prepare.py): fixed benchmark harness. It samples the prior, solves Burgers with a deterministic finite-volume SSP-RK3 solver, caches the dataset, and owns the evaluation metric.
- [`train.py`](train.py): the only editable experiment surface.
- [`program.md`](program.md): the autoresearch protocol.

## Tracked Benchmark

The tracked repo code currently defines the default small benchmark:

- domain: `x in [-1, 1]`, `t in [0, 1]`
- PDE: `u_t + u u_x - nu u_xx = 0`
- viscosity prior: log-uniform on `[2.5e-3, 5.0e-2]`
- initial-condition prior: hierarchical Gaussian random field with sampled amplitude, lengthscale, and bias
- dataset split: `512` train, `8` validation, `8` test
- grids: `128` initial-condition points and a `65 x 128` solution field
- experiment budget: `900` seconds of training per run

The cached dataset for the tracked benchmark lives under `~/.cache/autoresearch-burgers/surrogate/`.

The baseline in [`train.py`](train.py) is an Equinox/JAX DeepONet-style surrogate:

- a branch network processes the discretized initial condition plus viscosity
- a trunk network processes space-time coordinates
- their latent interaction reconstructs the field
- training uses pointwise supervision plus an auxiliary `t=0` consistency loss

The keep/discard metric is always **`val_rel_l2`** on the fixed validation split. Test metrics are reported for context only.

## Large-Run Results

We also ran a larger isolated cluster experiment with a much harder split:

- `100000` train
- `20000` validation
- `500` test

That large run was executed from an isolated temp clone so the tracked repo code stayed stable while the autonomous search loop ran on Gautschi. The committed artifact bundle here captures the outputs from that finished run:

- ledger copy: [`artifacts/2026-03-23-large-run/results.tsv`](artifacts/2026-03-23-large-run/results.tsv)
- machine-readable summary: [`artifacts/2026-03-23-large-run/summary.json`](artifacts/2026-03-23-large-run/summary.json)
- progress figure: [`artifacts/2026-03-23-large-run/figure_progress.svg`](artifacts/2026-03-23-large-run/figure_progress.svg)
- baseline vs best field examples: [`artifacts/2026-03-23-large-run/figure_examples_fields.svg`](artifacts/2026-03-23-large-run/figure_examples_fields.svg)
- baseline vs best slice comparisons: [`artifacts/2026-03-23-large-run/figure_examples_slices.svg`](artifacts/2026-03-23-large-run/figure_examples_slices.svg)
- TikZ loop diagram source and PDF:
  - [`artifacts/2026-03-23-large-run/autoresearch_loop_tikz.tex`](artifacts/2026-03-23-large-run/autoresearch_loop_tikz.tex)
  - [`artifacts/2026-03-23-large-run/autoresearch_loop_tikz.pdf`](artifacts/2026-03-23-large-run/autoresearch_loop_tikz.pdf)

Headline result from that large run:

- baseline: `val_rel_l2 = 2.963700e-02`
- best run: `val_rel_l2 = 1.683636e-02`
- relative improvement: `1.76x`
- validation error reduction: `43.19%`

The best run in that 20-experiment search was experiment `17`, commit `a37ff71`, described in the ledger as `widen branch and trunk hidden layers to 640`.

## Reproducing Figures

The figure generator is committed as [`scripts/make_large_run_figures.py`](scripts/make_large_run_figures.py).

It expects a fetched large-run workspace with the cluster artifacts already mirrored locally. Example:

```bash
python3 scripts/make_large_run_figures.py \
  --workspace /tmp/autoresearch-codex-large.woja6k \
  --output-dir artifacts/2026-03-23-large-run
```

This script reads the fetched checkpoint bundles, cross-checks every `results.tsv` row against the saved checkpoint metadata, and regenerates the committed figures.

## Local Usage

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and enough compute for JAX training.

```bash
uv sync
uv run prepare.py --jobs 8
uv run train.py
```

`uv` keeps dependencies in the repo-local `.venv/` instead of the global environment.

## Cluster Usage

For heavier runs, use the `cluster-slurm` skill and the workflow described in [`program.md`](program.md):

- build the cached dataset on CPU
- run training on GPU
- download checkpoints and logs after every completed run

The lightweight `python3 prepare.py --help > /dev/null` command in GPU workloads is intentional. The cluster runner auto-uploads Python scripts referenced in workload commands, which guarantees that `prepare.py` is staged next to `train.py` for the remote `import prepare`.

## License

MIT
