# autoresearch

This repo adapts the autoresearch loop to a scientific machine learning problem:
JAX-based physics-informed neural networks for the 1D viscous Burgers equation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example `mar22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read the in-scope files**: the repo is still intentionally small. Read:
   - `README.md` — repository context and workflow.
   - `prepare.py` — fixed Burgers problem definition, reference artifact generation, sampling utilities, and evaluation metric. Do not modify.
   - `train.py` — the only file you modify. JAX network family, optimizer phases, loss weights, collocation sizes, and training strategy all live here.
4. **Verify reference artifacts exist**: check that `~/.cache/autoresearch-burgers/reference/manifest.json` and `~/.cache/autoresearch-burgers/reference/burgers_reference.npz` exist. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: create `results.tsv` with only the header row shown below. The baseline is logged after the first run.
6. **Confirm and go**: once the cache and branch are ready, confirm setup looks good.

Once you get confirmation, begin the experiment loop.

## Experimentation

Each experiment runs for a **fixed 5-minute time budget**. Launch it as:

```bash
uv run train.py
```

**What you CAN do:**
- Modify `train.py` only.
- Change the neural representation in `train.py`: plain MLP, residual MLP, SIREN, input encoding, depth, width, activation, etc.
- Change the training algorithm in `train.py`: optimizer phases, learning-rate schedule, loss weights, sampling strategy, collocation counts, gradient clipping, and related hyperparameters.

**What you CANNOT do:**
- Modify `prepare.py`. It is the fixed benchmark harness.
- Modify the cached reference solution or the validation grid.
- Install new packages or add dependencies.
- Change the evaluation metric. `evaluate_model` in `prepare.py` is ground truth.

## Objective

**Minimize `val_rel_l2`. Lower is better.**

This is the relative L2 error of the learned Burgers solution on the fixed validation grid built in `prepare.py`.

Secondary metrics such as `val_rmse`, `val_max_abs`, and memory are for context only. Keep changes simple when possible: if two ideas perform similarly, prefer the simpler code path.

The first run must always establish the baseline using the current `train.py`.

## Output format

Each run ends with a summary like:

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

Example:

```text
commit	val_rel_l2	memory_gb	status	description
a1b2c3d	1.234567e-02	0.8	keep	baseline tanh mlp
b2c3d4e	9.876543e-03	1.0	keep	switch to Fourier features
c3d4e5f	1.410000e-02	0.8	discard	reduce interior points too far
d4e5f6g	0.000000e+00	0.0	crash	unstable SGD phase
```

## Experiment loop

The experiment runs on a dedicated branch such as `autoresearch/mar22`.

LOOP FOREVER:

1. Inspect the current branch and commit.
2. Change `train.py` with one clear experimental idea.
3. Commit the change.
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_rel_l2:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, diagnose, and decide whether the idea should be fixed or discarded.
7. Record the result in `results.tsv` (leave this file untracked).
8. If `val_rel_l2` improved, keep the commit and continue from there.
9. If `val_rel_l2` is equal or worse, reset to the previous good commit.

**Timeout**: if a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: fix obvious mistakes if the underlying idea still seems sound. If the idea itself is bad or unstable, log `crash`, revert, and move on.

**Never stop**: once the loop begins, do not pause to ask the human whether to continue. Keep iterating until interrupted.
