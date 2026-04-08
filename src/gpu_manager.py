#!/usr/bin/env python3
"""
GPU Manager Orchestrator for PyMARL Autoresearch

Monitors GPU availability, launches experiments in parallel, tracks metrics,
and performs early termination when experiments are clearly underperforming.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import yaml
from tensorboard.backend.event_processing import event_accumulator


@dataclass
class Experiment:
    """Tracks a single experiment run."""
    exp_id: str
    tag: str
    config: dict
    gpu_id: int
    process: subprocess.Popen = None
    start_time: float = field(default_factory=time.time)
    last_metric: dict = field(default_factory=dict)
    stage: str = 'pending'  # pending, stage1, stage2, stage3, done, killed, crashed
    kill_reason: str = ''
    # 用于计算 stage3_drop_ratio 和 sample_efficiency
    stage2_end_win_rate: float = 0.0
    last_time_above_baseline_90: int = 0  # 最后一次高于 baseline 90% 的 t_env

    def get_latest_metrics(self) -> Optional[dict]:
        """Read Sacred info.json to get latest metrics."""
        run_dir = Path('results/sacred')
        if not run_dir.exists():
            return None

        # Find the run directory matching our exp_id
        # Sacred run_id could be string or int, normalize to string for comparison
        exp_id_str = str(self.exp_id)
        for run_path in run_dir.iterdir():
            if run_path.name == exp_id_str:
                info_path = run_path / 'info.json'
                if info_path.exists():
                    try:
                        return json.loads(info_path.read_text())
                    except json.JSONDecodeError:
                        return None
        return None

    def get_t_env(self) -> int:
        """Get current t_env from metrics."""
        metrics = self.get_latest_metrics()
        if metrics is None:
            return 0
        return get_last_value(metrics, 't_env') or 0

    def get_battle_won(self) -> float:
        """Get current test_battle_won_mean from metrics."""
        metrics = self.get_latest_metrics()
        if metrics is None:
            return 0.0
        return get_last_value(metrics, 'test_battle_won_mean') or 0.0

    def determine_stage(self) -> str:
        """Determine current training stage based on t_env and config."""
        if self.stage in ('done', 'killed', 'crashed'):
            return self.stage

        t_env = self.get_t_env()
        q_tot_steps = self.config.get('q_tot_stage_steps', 500000)
        reward_steps = self.config.get('reward_stage_steps', 0)
        t_max = self.config.get('t_max', q_tot_steps + reward_steps)

        stage2_end = q_tot_steps + reward_steps

        if t_env < q_tot_steps:
            return 'stage1'
        elif t_env < stage2_end:
            return 'stage2'
        elif t_env < t_max:
            return 'stage3'
        else:
            return 'done'


def get_last_value(metrics: dict, key: str):
    """Extract last value from a metric list."""
    if key not in metrics:
        return None
    v = metrics[key]
    if isinstance(v, list) and len(v) > 0:
        return float(v[-1])
    return None


def load_baseline_curve(baseline_dir: str) -> dict:
    """构建 baseline 学习曲线，按 t_env 索引，取 5 次 run 的中位数。

    Args:
        baseline_dir: 包含多个 baseline run 目录的父目录

    Returns:
        dict: {t_env: median_win_rate}
    """
    from collections import defaultdict

    # 收集所有 run 的 (t_env, win_rate) 数据
    all_runs = defaultdict(list)  # t_env -> [win_rate1, win_rate2, ...]

    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        print(f"WARNING: Baseline directory not found: {baseline_dir}")
        return {}

    for run_dir in baseline_path.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()
            win_events = ea.Scalars('test_battle_won_mean')
            # e.step 就是 t_env
            for e in win_events:
                all_runs[int(e.step)].append(e.value)
        except Exception as e:
            print(f"WARNING: Failed to read {run_dir}: {e}")

    # 每个 t_env 取中位数
    curve = {}
    for t_env, wins in sorted(all_runs.items()):
        curve[t_env] = statistics.median(wins)

    print(f"Loaded baseline curve with {len(curve)} time points")
    return curve


def get_gpu_status() -> Dict[int, dict]:
    """Query nvidia-smi for GPU status.

    Returns:
        Dict mapping gpu_id to {'free_mb': int, 'used_mb': int, 'util': int, 'busy': bool}
        busy is True if memory > 20000MiB OR GPU-Util > 90%
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: nvidia-smi not found or failed. Assuming no GPUs available.")
        return {}

    gpu_status = {}
    for line in result.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            gpu_id = int(parts[0])
            used_mb = int(parts[1])
            util = int(parts[2].rstrip('%'))  # Remove trailing %
            # GPU is busy if memory > 20GB OR GPU-Util > 90%
            busy = used_mb > 20000 or util > 90
            gpu_status[gpu_id] = {
                'used_mb': used_mb,
                'util': util,
                'busy': busy
            }
    return gpu_status


def get_free_gpus(min_free_mb: int = 8000) -> List[int]:
    """Get list of free GPU IDs with sufficient free memory.

    Returns:
        List of available gpu_ids
    """
    status = get_gpu_status()
    return [
        gpu_id for gpu_id, info in status.items()
        if not info['busy'] and (24000 - info['used_mb']) >= min_free_mb
    ]


def should_terminate_early(exp: Experiment, baseline_curve: dict) -> Tuple[bool, str]:
    """Determine if an experiment should be terminated early.

    Early termination logic based on 3-stage training and time-matched baseline comparison:
    - Stage 1/2: Algorithm is equivalent to QMIX, should not perform terribly
    - Stage 3: Real test of the algorithm, allow transition period

    Args:
        exp: The experiment to check
        baseline_curve: dict mapping t_env to median baseline win rate

    Returns:
        Tuple of (should_terminate, reason)
    """
    if not baseline_curve:
        return False, ''

    t_env = exp.get_t_env()
    won = exp.get_battle_won()

    # 找到最接近的 baseline t_env
    baseline_t_envs = sorted(baseline_curve.keys())
    if not baseline_t_envs:
        return False, ''

    closest_t = min(baseline_t_envs, key=lambda x: abs(x - t_env))

    # 如果 t_env 差距太大（超过 100K），用最后可用的 baseline 值
    if abs(t_env - closest_t) > 100000 and baseline_t_envs:
        closest_t = baseline_t_envs[-1]

    baseline_at_t = baseline_curve.get(closest_t, 0)
    if baseline_at_t <= 0:
        return False, ''

    q_tot_steps = exp.config.get('q_tot_stage_steps', 500000)
    reward_steps = exp.config.get('reward_stage_steps', 0)
    stage2_end = q_tot_steps + reward_steps

    if t_env < stage2_end:
        # Stage 1/2: < 50% of baseline at same t_env
        threshold = baseline_at_t * 0.5
        if won > 0 and won < threshold:
            return True, f"Stage1/2 win={won:.3f} < {threshold:.3f} (50% of baseline@t={closest_t})"
    else:
        # Stage 3: < 90% of baseline at same t_env for >200K steps
        threshold = baseline_at_t * 0.9
        stage3_steps = t_env - stage2_end
        if won > 0 and won < threshold and stage3_steps > 200000:
            return True, f"Stage3 win={won:.3f} < {threshold:.3f} (90% of baseline@t={closest_t}) for >200K"

    return False, ''


def get_baseline_final_90(baseline_curve: dict) -> float:
    """获取 baseline 最终胜率中位数的 90%。

    Args:
        baseline_curve: {t_env: median_win_rate}

    Returns:
        baseline 最终胜率中位数的 90%
    """
    if not baseline_curve:
        return 0.0
    sorted_items = sorted(baseline_curve.items())
    final_median = sorted_items[-1][1] if sorted_items else 0.0
    return final_median * 0.9


def build_command(exp: Experiment) -> str:
    """Build the python command to launch an experiment.

    Args:
        exp: The experiment configuration

    Returns:
        Command string to execute
    """
    base_cmd = 'conda run -n hyr2pymarl python3 -u src/main.py'

    # Core config
    config = exp.config.get('config', 'qmix')
    env_config = exp.config.get('env_config', 'sc2')
    cmd = f'{base_cmd} --config={config} --env-config={env_config}'

    # Tag
    if 'tag' in exp.config:
        cmd += f' --tag={exp.tag}'

    # Collect all with-styled args (env_args and overrides)
    with_args = []

    # Env args
    env_args = exp.config.get('env_args', {})
    for key, value in env_args.items():
        with_args.append(f"env_args.{key}={value}")

    # Overrides
    overrides = exp.config.get('overrides', {})
    for key, value in overrides.items():
        if isinstance(value, str):
            with_args.append(f"{key}={value}")
        elif isinstance(value, bool):
            with_args.append(f"{key}={str(value).lower()}")
        else:
            with_args.append(f"{key}={value}")

    if with_args:
        cmd += " with " + " ".join(with_args)

    return cmd


class ExperimentManager:
    """Manages the experiment queue and GPU allocation."""

    def __init__(self, args):
        self.args = args
        self.queue: List[dict] = []  # Queue of experiment configs
        self.running: Dict[int, Experiment] = {}  # gpu_id -> Experiment
        self.completed: List[Experiment] = []

        # Load baseline curve for early termination comparison
        self.baseline_curve = load_baseline_curve(args.baseline_dir)
        self.baseline_final_90 = get_baseline_final_90(self.baseline_curve)
        print(f"Baseline final 90% threshold: {self.baseline_final_90:.3f}")

        self.load_queue()

        os.makedirs('results/logs', exist_ok=True)

    def load_queue(self):
        """Load experiment queue from config file(s)."""
        for config_file in self.args.exp_configs:
            path = Path(config_file)
            if not path.exists():
                print(f"WARNING: Config file not found: {config_file}")
                continue

            with open(path) as f:
                data = yaml.safe_load(f)

            queue = data.get('queue', [])
            self.queue.extend(queue)

        print(f"Loaded {len(self.queue)} experiments into queue")

    def launch_experiment(self, exp_config: dict, gpu_id: int) -> Experiment:
        """Launch a single experiment on specified GPU."""
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f'_{gpu_id}'
        tag = exp_config.get('tag', 'unnamed')

        exp = Experiment(
            exp_id=exp_id,
            tag=tag,
            config=exp_config,
            gpu_id=gpu_id
        )

        cmd = build_command(exp)
        log_file = f'results/logs/{exp_id}.log'

        # Set CUDA_VISIBLE_DEVICES to isolate GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"[LAUNCH] GPU {gpu_id} | {tag} | cmd: {cmd}")

        with open(log_file, 'w') as f:
            f.write(f"# GPU Manager launched experiment\n")
            f.write(f"# GPU: {gpu_id}\n")
            f.write(f"# Tag: {tag}\n")
            f.write(f"# Command: {cmd}\n")
            f.write(f"# Start time: {datetime.now()}\n")
            f.write("=" * 80 + "\n")

        process = subprocess.Popen(
            ['bash', '-c', cmd],
            env=env,
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Wait for Sacred to output run ID and parse it
        sacred_run_id = None
        for _ in range(60):  # Wait up to 6 seconds (was 3)
            time.sleep(0.1)
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    # Sacred outputs: Started run with ID "123456" (with quotes)
                    match = re.search(r'Started run with ID "?(\d+)"?', content)
                    if match:
                        sacred_run_id = match.group(1)
                        break
            except Exception:
                pass

        if sacred_run_id:
            print(f"[INFO] Sacred run ID: {sacred_run_id}")
            # Store the actual Sacred run_id for metrics lookup
            exp.exp_id = sacred_run_id
        else:
            # Fallback: scan results/sacred/ for the latest directory
            print(f"[WARNING] Could not parse Sacred run ID, scanning for latest run...")
            sacred_dir = Path('results/sacred')
            if sacred_dir.exists():
                run_dirs = [d for d in sacred_dir.iterdir() if d.is_dir()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
                    exp.exp_id = latest_run.name
                    print(f"[INFO] Using latest Sacred run: {exp.exp_id}")
                else:
                    print(f"[WARNING] No Sacred runs found, using exp_id: {exp_id}")
            else:
                print(f"[WARNING] Sacred directory not found, using exp_id: {exp_id}")

        exp.process = process
        exp.stage = 'stage1'  # Initial stage
        return exp

    def check_gpu_availability(self):
        """Check which GPUs are available."""
        return get_free_gpus(min_free_mb=8000)

    def launch_if_possible(self):
        """Launch new experiments if GPUs and queue are available."""
        free_gpus = self.check_gpu_availability()

        # Remove GPUs that have running experiments
        occupied_gpus = set(self.running.keys())
        available_gpus = [g for g in free_gpus if g not in occupied_gpus]

        # Launch as many experiments as we can
        while available_gpus and self.queue:
            gpu_id = available_gpus.pop(0)
            exp_config = self.queue.pop(0)

            exp = self.launch_experiment(exp_config, gpu_id)
            self.running[gpu_id] = exp
            print(f"[STATUS] Running: {len(self.running)}, Queued: {len(self.queue)}")

    def check_running_experiments(self):
        """Check status of all running experiments, handle early termination."""
        for gpu_id, exp in list(self.running.items()):
            # Update stage
            prev_stage = exp.stage
            exp.stage = exp.determine_stage()

            # Track stage2_end_win_rate when entering stage3
            if prev_stage in ('stage1', 'stage2') and exp.stage == 'stage3':
                exp.stage2_end_win_rate = exp.get_battle_won()

            # Check if process died
            if exp.process and exp.process.poll() is not None:
                exp.stage = 'crashed'
                self.completed.append(exp)
                del self.running[gpu_id]
                print(f"[ENDED] {exp.tag} crashed on GPU {gpu_id}")
                continue

            # Check if experiment completed normally
            if exp.stage == 'done':
                self.completed.append(exp)
                del self.running[gpu_id]
                print(f"[DONE] {exp.tag} completed on GPU {gpu_id}")
                continue

            # Track last_time_above_baseline_90 for sample_efficiency
            won = exp.get_battle_won()
            if won > 0 and self.baseline_final_90 > 0:
                if won >= self.baseline_final_90:
                    exp.last_time_above_baseline_90 = exp.get_t_env()

            # Check for early termination
            should_stop, reason = should_terminate_early(exp, self.baseline_curve)
            if should_stop:
                print(f"[EARLY_STOP] {exp.tag} on GPU {gpu_id}: {reason}")
                self.kill_experiment(exp)
                exp.kill_reason = reason
                self.completed.append(exp)
                del self.running[gpu_id]

    def kill_experiment(self, exp: Experiment):
        """Kill a running experiment."""
        if exp.process and exp.process.poll() is None:
            exp.process.terminate()
            try:
                exp.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                exp.process.kill()
        exp.stage = 'killed'
        print(f"[KILLED] {exp.tag} (PID {exp.process.pid if exp.process else 'N/A'})")

    def report_progress(self):
        """Print current status of all experiments."""
        if not self.running:
            return

        now = datetime.now()
        print(f"\n{'='*80}")
        print(f"[{now.strftime('%Y-%m-%d %H:%M')}] {len(self.running)} experiments running:")
        for gpu_id, exp in sorted(self.running.items()):
            elapsed_min = (time.time() - exp.start_time) / 60
            t_env = exp.get_t_env()
            won = exp.get_battle_won()

            # Format t_env nicely
            if t_env >= 1000000:
                t_str = f"{t_env/1000000:.2f}M"
            elif t_env >= 1000:
                t_str = f"{t_env/1000:.0f}K"
            else:
                t_str = str(t_env)

            print(f"  GPU {gpu_id} | {exp.tag:20s} | {elapsed_min:5.1f}min | {exp.stage:7s} | "
                  f"t_env={t_str:>8s} | won={won:.3f}")

        # Also show queue status
        if self.queue:
            print(f"  Queue: {len(self.queue)} experiments waiting")
        print(f"{'='*80}\n")

    def cleanup_finished(self):
        """Clean up finished experiments and log results."""
        for exp in self.completed:
            # Write to results.tsv
            self.log_result(exp)

        # Clear completed list
        self.completed.clear()

    def log_result(self, exp: Experiment):
        """Log experiment result to results.tsv."""
        results_file = Path('results.tsv')
        won = exp.get_battle_won()

        # Compute stage3_drop_ratio
        if exp.stage2_end_win_rate > 0:
            stage3_drop_ratio = (exp.stage2_end_win_rate - won) / exp.stage2_end_win_rate
        else:
            stage3_drop_ratio = None

        # Compute sample_efficiency: last t_env when win_rate was >= baseline 90%
        # If never above 90%, record -1
        if exp.last_time_above_baseline_90 > 0:
            sample_efficiency = exp.last_time_above_baseline_90
        else:
            sample_efficiency = -1

        # Format: commit, test_battle_won_mean, stage3_drop_ratio, sample_efficiency, status, description
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            encoding='utf-8'
        ).strip()

        drop_str = f"{stage3_drop_ratio:.3f}" if stage3_drop_ratio is not None else "na"
        eff_str = f"{sample_efficiency}" if sample_efficiency > 0 else "-1"

        row = f"{commit}\t{won:.6f}\t{drop_str}\t{eff_str}\t{exp.stage}\t{exp.tag}\n"

        with open(results_file, 'a' if results_file.exists() else 'w') as f:
            if not results_file.exists():
                f.write("commit\ttest_battle_won_mean\tstage3_drop_ratio\tsample_efficiency\tstatus\ttag\n")
            f.write(row)

        print(f"[LOGGED] {exp.tag}: won={won:.6f}, drop={drop_str}, eff={eff_str}, status={exp.stage}")

    def run(self):
        """Main loop."""
        print("=" * 80)
        print("GPU Manager starting...")
        print(f"Queue size: {len(self.queue)}")
        print(f"Check interval: {self.args.check_interval}s")
        print("=" * 80)

        # Initial launch
        self.launch_if_possible()

        while self.running or self.queue:
            time.sleep(self.args.check_interval)

            self.launch_if_possible()
            self.check_running_experiments()
            self.report_progress()
            self.cleanup_finished()

        print("All experiments completed. GPU Manager exiting.")


def main():
    parser = argparse.ArgumentParser(
        description='GPU Manager Orchestrator for PyMARL Autoresearch'
    )
    parser.add_argument(
        '--exp_configs',
        nargs='+',
        required=True,
        help='Experiment queue config YAML file(s)'
    )
    parser.add_argument(
        '--baseline_dir',
        default='results/baseline/MMM2',
        help='Path to QMIX baseline runs directory (default: results/baseline/MMM2)'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=1800,
        help='GPU check interval in seconds (default: 1800 = 30 min)'
    )

    args = parser.parse_args()

    manager = ExperimentManager(args)
    manager.run()


if __name__ == '__main__':
    main()
