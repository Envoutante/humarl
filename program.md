# pymarl-autoresearch

This protocol adapts the autonomous experiment loop to the PyMARL codebase.

## GPU Manager Orchestrator

For parallel experiment execution and automated monitoring, use the GPU manager:

```bash
./run_gpu_manager.sh gpu_manager.log \
   --exp_configs src/config/autoresearch.yaml \
   --baseline_dir results/baseline/MMM2 \
   --check_interval 300
```

**What it does**:
- Monitors GPU availability via `nvidia-smi` (checks every 30 minutes by default)
- Launches experiments on free GPUs automatically
- Tracks metrics by reading Sacred `info.json` periodically
- Reports progress every check interval
- Performs early termination when experiments clearly underperform

### Early Termination Rules

Based on the 3-stage training structure and time-matched QMIX baseline comparison:

| Stage | Condition | Action |
|-------|-----------|--------|
| Stage 1/2 | `win_rate[t_env] < baseline[t_env] * 0.5` | Terminate (abnormally poor, likely a bug) |
| Stage 3 | `win_rate[t_env] < baseline[t_env] * 0.9` for >200K steps | Terminate (clearly below baseline) |

**Baseline source**: 5 QMIX runs under `results/baseline/MMM2/`, used to build the learning curve `{t_env: median_win_rate}`. Thresholds are compared against the baseline median at the same timestep.

**Threshold rationale**: Stage 1/2 uses 50% because the method should behave like QMIX at this point, so major deviations indicate issues. Stage 3 uses 90% because your method typically performs well on MMM2.

## Setup

**Conda environment**: PyMARL must run in the `hyr2pymarl` conda environment. Activate env first, then run `python3` directly.

To start a new run, work with the user to:

1. **Agree on a run tag**: propose a date-based tag (e.g. `mar26`) and use branch `autoresearch/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master/main state.
3. **Read the in-scope files** for full context:
   - `README.md` — project context and canonical run command.
   - `src/main.py` — experiment entrypoint, Sacred observer setup, CLI/config merge.
   - `src/config/default.yaml` — global defaults (including `t_max`).
   - `src/config/algs/qmix.yaml` — baseline algorithm config.
   - `src/config/envs/sc2.yaml` — SC2 environment config.
   - `src/gpu_manager.py` — orchestrator script (if using parallel mode).
4. **Verify runtime dependencies**:
   - SC2/SMAC setup must be available (see `install_sc2.sh`).
   - Python dependencies from `requirements.txt` must already be installed.
5. **Initialize `results.tsv`** in repo root with header only. Baseline is logged after first run.
6. **First run policy**: Always run an untouched baseline first before any tuning.
7. **Confirm and begin** the experiment loop.

## Experimentation

Each experiment runs on a single GPU.

**Budget rule**: use a fixed `t_max` (from config/overrides) for fair comparisons. Do not use fixed wall-clock time as the primary budget.

**Canonical baseline command**:

```bash
python3 -u src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

**Primary objective**: maximize `test_battle_won_mean` (higher is better).

**Tie-breaker**: if primary metric is effectively tied, prefer better `sample_efficiency`, then lower complexity.

### Current research context

Current work is a QMIX-based reward decomposition variant targeting better Bellman consistency for `Q_i`.

Default experiment template (adapt as needed, but treat as the starting point):

```bash
nohup python3 -u src/main.py \
   --tag=res_reg_multi \
   --config=qmix \
   --env-config=sc2 \
   with env_args.map_name=MMM2 \
       t_max=2100000 \
       reward_mixer=True \
       reward_prediction_mode=residual \
       reg_weight=0.5 \
       q_tot_stage_steps=1200000 \
       reward_stage_steps=300000 \
   &> output.out &
```

### Stage3 transition risk

Stage3 transition drop is a known failure mode in this project.
Treat it as a first-priority diagnosis target, not as normal variance noise.

### Diagnosis and keep/discard rules

For every run, explicitly compare metrics before and after the stage3 switch:

- `test_battle_won_mean` (primary)
- `stage3_drop_ratio` (computed automatically)
- `sample_efficiency` (computed automatically)

Decision rules:

- If final performance improves but stage3 drop becomes clearly worse, mark the run as a risk candidate and default to `discard`.
- If final performance is tied, prefer the run with a smoother stage3 transition (smaller drop and faster recovery).
- Keep a run only when final gain and stage3 behavior are jointly acceptable.

### Scope policy (strict whitelist)

Core algorithm files that may be edited:

- `src/learners/q_learner.py`
- `src/modules/reward_mixer.py`
- `src/config/default.yaml`

Infrastructure files (orchestrator only, not algorithm logic):

- `src/gpu_manager.py`
- `src/config/autoresearch.yaml`

### Not allowed

- Editing any non-whitelist file (especially `src/runners/**`, `src/envs/**`, `src/main.py`, `third_party/**`)
- Installing new packages or changing external dependencies

### First run policy

Always run an untouched baseline first before any tuning.

### Simplicity criterion

If two variants are close, keep the simpler one.
Small gains that add brittle complexity should usually be discarded.

## Output and metric extraction

PyMARL writes Sacred runs to `results/sacred/<run_id>/`.
`gpu_manager.py` is the source of truth for metric tracking and status updates.
Use `results.tsv` as the primary metrics table, and use `results/logs/` plus
`results/sacred/<run_id>/` only for debugging and audit when needed.

## Logging results

Log every completed attempt to `results.tsv` (tab-separated, not comma-separated).

Header:

```tsv
commit	test_battle_won_mean	stage3_drop_ratio	sample_efficiency	status	tag
```

Columns:

1. short git commit hash (7 chars)
2. best/final `test_battle_won_mean` for the run (use `0.000000` for crashes)
3. `stage3_drop_ratio`: (pre-stage3 win rate - post-stage3 win rate) / pre-stage3 win rate; use `na` for crash/early_stop
4. `sample_efficiency`: the `t_env` required to reach 90% of the baseline final value; use `-1` if never reached
5. status in `{stage1, stage2, stage3, done, killed, crashed}`
6. experiment tag/identifier

Example:

```tsv
commit	test_battle_won_mean	stage3_drop_ratio	sample_efficiency	status	tag
a1b2c3d	0.421875	0.050	1800000	done	res_reg_v1
b2c3d4e	0.468750	0.120	-1	killed	res_reg_v2
c3d4e5f	0.406250	na	na	crashed	res_reg_v3
```

Do not commit `results.tsv`.

## Experiment loop

Run on a dedicated branch such as `autoresearch/mar26`.

Use this loop for autonomous algorithm iteration and execution.

LOOP FOREVER:

1. Check the current branch and commit; keep work on `autoresearch/<tag>`.
2. Read `results.tsv` and recent logs to identify the current best variant and the biggest failure mode (especially stage3 drop).
3. Propose a small, concrete next hypothesis for algorithm improvement.
4. Edit only allowed algorithm files (`src/learners/q_learner.py`, `src/modules/reward_mixer.py`, `src/config/default.yaml`).
5. Commit the candidate change.
6. Add one or more experiments for this hypothesis to `src/config/autoresearch.yaml`.
7. Launch or keep running GPU manager:

```bash
./run_gpu_manager.sh gpu_manager.log \
   --exp_configs src/config/autoresearch.yaml \
   --baseline_dir results/baseline/MMM2 \
   --check_interval 300
```

8. Let GPU manager schedule experiments in parallel, monitor Sacred metrics, apply early termination rules, and append outcomes to `results.tsv`.
9. After enough signal is collected, compare candidates by primary metric (`test_battle_won_mean`) and stage3 behavior (`stage3_drop_ratio`, `sample_efficiency`).
10. Keep the best commit as new baseline for the next iteration; discard regressions by resetting to the pre-experiment commit.
11. Repeat without pausing unless explicitly interrupted by the human.

Operational notes:
- Treat crash or missing metrics as failure and log them explicitly.
- Prefer smaller, interpretable changes over large coupled edits.
- If two variants are close, keep the simpler one.

## Failure handling

- **Crash (OOM/bug/config error)**: mark as `crash` if not quickly fixable within whitelist files.
- **Timeout**: if a run exceeds a reasonable envelope for the chosen `t_max` (for example, >2x normal runtime on same machine), stop it, mark as failure, and revert.

## Autonomy rule

After setup is complete and baseline is established, keep iterating without pausing for confirmation between experiments.
Stop only when explicitly interrupted by the human.
