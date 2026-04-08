# pymarl-autoresearch

This protocol adapts the autonomous experiment loop to the PyMARL codebase.

## GPU Manager Orchestrator (Optional)

For parallel experiment execution and automated monitoring, use the GPU manager:

```bash
conda run -n hyr2pymarl python3 src/gpu_manager.py \
   --exp_configs src/config/autoresearch.yaml \
   --check_interval 1800
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
| Stage 1/2 | `win_rate[t_env] < baseline[t_env] * 0.5` | Terminate (异常差，可能有 bug) |
| Stage 3 | `win_rate[t_env] < baseline[t_env] * 0.9` for >200K steps | Terminate (明显低于 baseline) |

**Baseline 来源**: `results/baseline/MMM2/` 下的 5 次 QMIX run，构建学习曲线 `{t_env: median_win_rate}`，阈值与同一时间步的 baseline 中位数对比。

**阈值说明**: Stage 1/2 用 50% 因为此时算法等同于 QMIX，任何明显偏差都说明有问题。Stage 3 用 90% 因为你的算法在 MMM2 上通常表现较好。

## Setup

**Conda environment**: PyMARL must run in the `hyr2pymarl` conda environment. All `python3` and `gpu_manager.py` commands must be wrapped with `conda run -n hyr2pymarl`.

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
conda run -n hyr2pymarl python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

**Primary objective**: maximize `test_battle_won_mean` (higher is better).

**Tie-breaker**: if primary metric is effectively tied, prefer better `sample_efficiency`, then lower complexity.

### Current research context

Current work is a QMIX-based reward decomposition variant targeting better Bellman consistency for `Q_i`.

Default experiment template (adapt as needed, but treat as the starting point):

```bash
nohup conda run -n hyr2pymarl python3 -u src/main.py \
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
       log_individual_reward=True \
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

For the latest run, extract metrics from `info.json`:

```bash
latest=$(ls -1 results/sacred | grep -E '^[0-9]+$' | sort -n | tail -1)
python3 - <<'PY'
import json
from pathlib import Path

runs = sorted([p for p in Path('results/sacred').iterdir() if p.name.isdigit()], key=lambda p: int(p.name))
info = json.loads((runs[-1] / 'info.json').read_text())

def last_value(name):
    v = info.get(name, [])
    return float(v[-1]) if isinstance(v, list) and len(v) else None

print(f"run_id={runs[-1].name}")
print(f"test_battle_won_mean={last_value('test_battle_won_mean')}")
PY
```

If a run crashes before producing valid Sacred metrics, inspect the log file in `results/logs/`.

## Logging results

Log every completed attempt to `results.tsv` (tab-separated, not comma-separated).

Header:

```tsv
commit	test_battle_won_mean	stage3_drop_ratio	sample_efficiency	status	tag
```

Columns:

1. short git commit hash (7 chars)
2. best/final `test_battle_won_mean` for the run (use `0.000000` for crashes)
3. `stage3_drop_ratio`: (stage3前胜率 - stage3后胜率) / stage3前胜率；如果是 crash/early_stop 则为 `na`
4. `sample_efficiency`: 达到 baseline 终值 90% 所需的 t_env；若未达到则为 `-1`
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

### Manual Loop (Single experiment at a time)

LOOP FOREVER:

1. Check current git branch and commit.
2. Propose one concrete PyMARL experiment idea.
3. Edit only `src/learners/q_learner.py`, `src/modules/reward_mixer.py`, and `src/config/default.yaml`.
4. Commit the change.
5. Run experiment:

```bash
conda run -n hyr2pymarl python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2
```

6. Parse latest Sacred metrics (`test_battle_won_mean`).
7. Run a stage3 drop check and record `stage3_drop_ratio`.
8. If metrics are missing, treat as crash.
9. Append one row to `results.tsv` with format: `commit test_battle_won_mean stage3_drop_ratio sample_efficiency status tag`.
10. If primary metric improves and stage3 behavior is acceptable, keep commit and advance branch.
11. If metric is tied/worse, or stage3 degradation is materially worse, reset to pre-experiment commit.

### Orchestrated Loop (Parallel experiments)

When using `gpu_manager.py`:

1. Define experiments in `src/config/autoresearch.yaml`.
2. Launch the orchestrator:

```bash
conda run -n hyr2pymarl python3 src/gpu_manager.py --exp_configs src/config/autoresearch.yaml --check_interval 1800
```

3. Orchestrator automatically:
   - Scans for free GPUs
   - Launches experiments from queue
   - Monitors metrics every 30 minutes
   - Reports progress
   - Performs early termination when needed
   - Logs results to `results.tsv`

4. Periodically review `results.tsv` and `results/logs/` for experiment outputs.

## Failure handling

- **Crash (OOM/bug/config error)**: mark as `crash` if not quickly fixable within whitelist files.
- **Timeout**: if a run exceeds a reasonable envelope for the chosen `t_max` (for example, >2x normal runtime on same machine), stop it, mark as failure, and revert.

## Autonomy rule

After setup is complete and baseline is established, keep iterating without pausing for confirmation between experiments.
Stop only when explicitly interrupted by the human.
