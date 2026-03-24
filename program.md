# autoresearch

This repo adapts the autoresearch loop to a harder scientific machine learning problem:
learning a Burgers surrogate operator over a family of viscosities and initial conditions.

The tracked benchmark on `main` is the large fixed split:

- `100000` train samples
- `20000` validation samples
- `500` test samples

The tracked `train.py` on `main` is the promoted post-search operator configuration from the finished large run. The first run on any new experiment branch still counts as that branch's baseline.

## Symmetry Agenda

This branch is specifically for **symmetry-aware operator discovery**.

The agent must work explicitly on finding an operator formulation that incorporates known symmetries, invariances, or equivariances of the fixed Burgers boundary value problem and satisfies them as automatically as possible by construction, not only by data exposure.

Examples of mechanisms the agent may explore inside `train.py`:

- reflection-aware or parity-aware operator formulations
- even/odd or symmetric/antisymmetric decompositions
- mirrored coordinate channels or tied weights
- invariant or equivariant latent representations
- canonicalized coordinates, normalized coordinates, or symmetry-respecting basis functions
- other operator parameterizations that hard-wire exact benchmark symmetries

Important constraint: do **not** assume a symmetry just because it exists for a related PDE on an infinite domain. The agent must reason explicitly about which symmetries survive the actual benchmark definition, including the finite interval, zero Dirichlet boundaries, viscosity parameterization, fixed evaluation grid, and hierarchical GP prior. Treat translation, rotation, and scaling symmetries as hypotheses to test against the benchmark, not as assumptions.

If a candidate symmetry is broken by the benchmark setup, record that and move on. The goal is to discover an operator formulation that respects the **actual** symmetries of this boundary value problem.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example `mar22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**:
   - `README.md` — repository context and workflow
   - `prepare.py` — fixed surrogate dataset harness and evaluation metric. Do not modify.
   - `train.py` — the only file you modify during experiments.
4. **Verify dataset artifacts exist**: check that `~/.cache/autoresearch-burgers/surrogate-large-100k20k500/manifest.json` and `~/.cache/autoresearch-burgers/surrogate-large-100k20k500/burgers_surrogate_dataset.npz` exist. If they do not exist locally and you intend to run on cluster, build them on the cluster first instead of blocking on a local cache.
5. **Initialize results.tsv**: create `results.tsv` with only the header row shown below. The baseline is logged after the first run.
6. **Confirm and go**: once the cache and branch are ready, confirm setup looks good and begin the loop.

## Experimentation

Each experiment runs for a **fixed 15-minute time budget**. Launch it as:

```bash
uv run train.py
```

For heavier runs, prefer the configured SLURM workflow:

- build `prepare.py` artifacts on CPU
- run `train.py` on GPU

For the current Gautschi setup, use:

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

The explicit `prepare.py --help` command in the GPU workload is part of the contract: the cluster runner auto-uploads Python scripts referenced in workload commands, and `train.py` imports `prepare.py`.

For cluster-backed runs, treat the printed `checkpoint_dir`, `checkpoint_meta`, and `prediction_file` as **remote paths** inside the cluster run directory. After every completed cluster run, download the checkpoint bundle back into this local repo, for example:

```bash
python3 ~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py download \
  --run-id <RUN_ID> \
  --remote-path results/checkpoints/<checkpoint-subdir> \
  --local-path results/checkpoints
```

When `--local-path` is an existing directory, the cluster downloader preserves the remote checkpoint folder name automatically.

Do not leave successful cluster experiments with their only checkpoint copy on the cluster scratch filesystem.

**What you CAN do:**

- Modify `train.py` only.
- Change the surrogate representation: branch/trunk network family, width, depth, latent size, coordinate encoding, and related architecture details.
- Change the operator formulation to encode exact symmetries or equivariances by construction: tied weights, mirrored features, invariant/equivariant coordinates, symmetry-aware latent bases, or canonicalization transforms.
- Change the training algorithm: optimizer phases, schedules, batch sizes, field-point sampling, loss functions, and auxiliary losses.

**What you CANNOT do:**

- Modify `prepare.py`. It is the fixed benchmark harness.
- Change the hierarchical GP prior, cached dataset artifacts, or evaluation split sizes.
- Install new packages or add dependencies during experiments.
- Change the keep/discard metric.
- Remove or silently bypass the checkpoint-writing contract in `train.py`.

## Objective

**Minimize `val_rel_l2`. Lower is better.**

This is the mean relative L2 error on the fixed validation split generated by `prepare.py`.

Secondary metrics such as `val_rmse`, `val_max_abs`, `val_worst_rel_l2`, and the reported test metrics are for context only. The first run must always establish the baseline using the current `train.py`.

For this branch, there is an additional scientific objective: search specifically for **symmetry-aware operator formulations** that improve validation error while respecting the actual invariances/equivariances of the benchmark by construction.

## Output format

Each run ends with a summary like:

```text
---
val_rel_l2:       1.234567e-02
val_rmse:         8.765432e-03
val_max_abs:      4.321000e-02
val_worst_rel_l2: 2.345678e-02
test_rel_l2:      1.456789e-02
test_rmse:        9.876543e-03
test_max_abs:     4.876000e-02
test_worst_rel_l2:3.456789e-02
training_seconds: 900.0
total_seconds:    907.5
peak_vram_mb:     812.4
num_steps:        3200
num_params_M:     0.842
model_family:     deeponet
branch_family:    resmlp
trunk_family:     mlp
coord_encoding:   fourier
optimizer_name:   adamw+adamw
checkpoint_dir:   /abs/path/to/results/checkpoints/<run-id>
checkpoint_meta:  /abs/path/to/results/checkpoints/<run-id>/metadata.json
prediction_file:  /abs/path/to/results/checkpoints/<run-id>/predictions.npz
```

To extract the main metric from a log file:

```bash
grep "^val_rel_l2:\|^peak_vram_mb:" run.log
```

## Logging results

Log every completed experiment to `results.tsv` (tab-separated, not comma-separated).

The TSV has 5 columns:

```text
commit	val_rel_l2	memory_gb	status	description
```

1. Git commit hash (short, 7 chars)
2. `val_rel_l2` achieved — use `0.000000e+00` for crashes
3. Peak memory in GB, rounded to one decimal place — use `0.0` for crashes
4. Status: `keep`, `discard`, or `crash`
5. Short experiment description

## Experiment loop

The experiment runs on a dedicated branch such as `autoresearch/mar22`.

LOOP FOREVER:

1. Inspect the current branch and commit.
2. State the next symmetry hypothesis clearly: what symmetry is being targeted, whether it is exact or approximate for this benchmark, and how the operator formulation is supposed to encode it.
3. Change `train.py` with one clear experimental idea.
4. Commit the change.
5. Run the experiment locally with `uv run train.py > run.log 2>&1`, or use the cluster recipe above for long runs.
6. Read results: `grep "^val_rel_l2:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, diagnose, and decide whether the idea should be fixed or discarded.
8. For cluster runs, download the remote checkpoint bundle named in the summary back into local `results/checkpoints/` before moving on.
9. Record the result in `results.tsv` (leave this file untracked). Make the description mention the symmetry idea explicitly.
10. If `val_rel_l2` improved, keep the commit and continue from there.
11. If `val_rel_l2` is equal or worse, revert to the previous good commit.

Each run also leaves an untracked checkpoint bundle in `results/checkpoints/`. Keep these artifacts. They are part of the traceability contract and include saved validation/test predictions for later analysis.

**Timeout**: if a run exceeds 25 minutes end to end, kill it and treat it as a failure.

**Crashes**: fix obvious mistakes if the underlying idea still seems sound. If the idea itself is bad or unstable, log `crash`, revert, and move on.

**Never stop**: once the loop begins, do not pause to ask the human whether to continue. Keep iterating until interrupted.
