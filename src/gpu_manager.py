#!/usr/bin/env python3
"""PyMARL 的 GPU 实验调度器。

负责监控 GPU 空闲状态、按队列拉起实验、持续追踪 Sacred 指标，
并在明显劣于 baseline 时触发早停。
"""

import argparse
import json
import threading
import os
import re
import statistics
import subprocess
import time
import smtplib
import sys
import atexit
from email.message import EmailMessage
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import yaml
from tensorboard.backend.event_processing import event_accumulator


# 通知配置（也可通过 --notification-email 覆盖接收地址）
# QQ 邮箱 SMTP 配置说明:
# 1. 登录 https://mail.qq.com -> 设置 -> 账户 -> POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务
# 2. 开启 "SMTP 服务" -> 生成授权码（不是 QQ 密码！）
# 3. 将授权码填入 SMTP_PASSWORD，将你的 QQ 邮箱填入 SMTP_USERNAME
NOTIFICATION_EMAIL = "1294352318@qq.com"  # 接收通知的邮箱地址
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465  # QQ 邮箱 SMTP 使用 SSL 加密
SMTP_USERNAME = "1294352318@qq.com"  # 你的 QQ 邮箱地址
SMTP_PASSWORD = "wzjuxwhokqdvffhi"  # 授权码（不是 QQ 登录密码！）
ALERT_LOG_FILE = "results/logs/alerts.log"


class TeeStream:
    """把同一份输出同时写入多个流。

    常用于把终端输出同步写入日志文件，并保证尽量实时刷新。
    """

    def __init__(self, *streams):
        """初始化输出流集合。"""
        self.streams = streams

    def write(self, data):
        """将文本写到所有流，并在每个流上立即 flush。"""
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                continue

    def flush(self):
        """刷新所有输出流。"""
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                continue

    def __getattr__(self, name):
        """将未知属性透传给第一个输出流。"""
        return getattr(self.streams[0], name)


def enable_manager_log(log_file: Optional[str]):
    """开启 manager 级别日志镜像。

    把当前进程的 stdout/stderr 同时写入指定文件，便于在后台运行时
    仍能稳定保留完整日志。
    """
    if not log_file:
        return

    log_path = Path(log_file)
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    fp = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def _restore_streams():
        """进程退出时恢复原始输出流并关闭日志文件句柄。"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass

    atexit.register(_restore_streams)

    sys.stdout = TeeStream(original_stdout, fp)
    sys.stderr = TeeStream(original_stderr, fp)

    print(f"[INFO] Manager tee log enabled: {log_path}", flush=True)


def send_notification(subject: str, body: str, email_to: str = None):
    """发送告警通知。

    先将告警内容追加到本地 alerts.log，再在 SMTP 配置可用时
    发送邮件给指定收件人。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_line = f"[{timestamp}] {subject}: {body}"

    # 始终写入本地告警日志
    os.makedirs("results", exist_ok=True)
    with open(ALERT_LOG_FILE, "a") as f:
        f.write(alert_line + "\n")

    print(f"[ALERT] {alert_line}", flush=True)

    # 如果配置了邮箱则发送邮件
    recipient = email_to or NOTIFICATION_EMAIL
    if recipient and SMTP_USERNAME and SMTP_PASSWORD:
        try:
            msg = EmailMessage()
            msg["From"] = SMTP_USERNAME
            msg["To"] = recipient
            msg["Subject"] = f"[PyMARL GPU Manager] {subject}"
            msg.set_content(body)

            # QQ 邮箱使用 SSL 加密（端口 465）
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"[NOTIFICATION] Email sent to {recipient}", flush=True)
        except Exception as e:
            print(f"[NOTIFICATION] Failed to send email: {e}", flush=True)


@dataclass
class Experiment:
    """单个实验运行时状态。"""

    exp_id: str
    tag: str
    config: dict
    gpu_id: int
    process: subprocess.Popen = None
    start_time: float = field(default_factory=time.time)
    last_metric: dict = field(default_factory=dict)
    stage: str = "pending"  # 取值：待启动、阶段1、阶段2、阶段3、完成、已杀死、已崩溃
    kill_reason: str = ""
    # 用于计算 stage3_drop_ratio 和 sample_efficiency
    stage2_end_win_rate: float = 0.0
    last_time_above_baseline_90: int = 0  # 最后一次高于 baseline 90% 的 t_env

    def get_latest_metrics(self) -> Optional[dict]:
        """读取当前实验的 Sacred 指标。

        会在 results/sacred 下按 exp_id 匹配目录，并尝试读取 info.json。
        """
        run_dir = Path("results/sacred")
        if not run_dir.exists():
            return None

        # 运行编号可能是字符串或整数，这里统一转成字符串比较
        exp_id_str = str(self.exp_id)
        for run_path in run_dir.iterdir():
            if run_path.name == exp_id_str:
                info_path = run_path / "info.json"
                if info_path.exists():
                    try:
                        return json.loads(info_path.read_text())
                    except json.JSONDecodeError:
                        return None
        return None

    def get_t_env(self) -> int:
        """返回当前实验训练步数 (t_env)。"""
        metrics = self.get_latest_metrics()
        if metrics is None:
            return 0
        return get_latest_t_env(metrics)

    def get_battle_won(self) -> float:
        """返回当前测试胜率 (test_battle_won_mean)。"""
        metrics = self.get_latest_metrics()
        if metrics is None:
            return 0.0
        return get_last_value(metrics, "test_battle_won_mean") or 0.0

    def determine_stage(self) -> str:
        """根据 t_env 判断当前处于哪一训练阶段。"""
        if self.stage in ("done", "killed", "crashed"):
            return self.stage

        t_env = self.get_t_env()
        q_tot_steps = self.config.get("q_tot_stage_steps", 500000)
        reward_steps = self.config.get("reward_stage_steps", 0)
        t_max = self.config.get("t_max", q_tot_steps + reward_steps)

        stage2_end = q_tot_steps + reward_steps

        if t_env < q_tot_steps:
            return "stage1"
        elif t_env < stage2_end:
            return "stage2"
        elif t_env < t_max:
            return "stage3"
        else:
            return "done"


def get_last_value(metrics: dict, key: str):
    """读取指标序列的最后一个有效数值。"""
    if key not in metrics:
        return None
    v = metrics[key]
    if isinstance(v, list) and len(v) > 0:
        try:
            return float(v[-1])
        except (TypeError, ValueError):
            return None
    return None


def get_latest_t_env(metrics: dict) -> int:
    """从 Sacred 的 episode_T 字段提取最新训练时间步。"""
    value = get_last_value(metrics, "episode_T")
    return int(value) if value is not None and value >= 0 else 0


def load_baseline_curve(baseline_dir: str) -> dict:
    """构建 baseline 学习曲线。

    读取 baseline 目录下多个 run 的 TensorBoard 事件，将相同时间步的
    test_battle_won_mean 聚合为中位数，并返回 {t_env: median_win_rate}。
    """
    from collections import defaultdict

    # 收集所有 run 的 (t_env, win_rate) 数据
    all_runs = defaultdict(list)  # 每个时间步对应多个胜率样本

    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        print(f"WARNING: Baseline directory not found: {baseline_dir}", flush=True)
        return {}

    for run_dir in baseline_path.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()
            win_events = ea.Scalars("test_battle_won_mean")
            # e.step 就是训练时间步
            for e in win_events:
                all_runs[int(e.step)].append(e.value)
        except Exception as e:
            print(f"WARNING: Failed to read {run_dir}: {e}", flush=True)

    # 每个 t_env 取中位数
    curve = {}
    for t_env, wins in sorted(all_runs.items()):
        curve[t_env] = statistics.median(wins)

    print(f"Loaded baseline curve with {len(curve)} time points", flush=True)
    return curve


def get_gpu_status() -> Dict[int, dict]:
    """查询每张 GPU 的显存和利用率状态。

    通过 nvidia-smi 返回 used_mb、util 和 busy 标记。
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "ERROR: nvidia-smi not found or failed. Assuming no GPUs available.",
            flush=True,
        )
        return {}

    gpu_status = {}
    for line in result.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpu_id = int(parts[0])
            used_mb = int(parts[1])
            util = int(parts[2].rstrip("%"))  # 去掉末尾百分号
            # 显存超过 20GB 或利用率超过 90% 视为繁忙
            busy = used_mb > 20000 or util > 90
            gpu_status[gpu_id] = {"used_mb": used_mb, "util": util, "busy": busy}
    return gpu_status


def get_free_gpus(min_free_mb: int = 8000) -> List[int]:
    """返回满足空闲条件的 GPU 列表。"""
    status = get_gpu_status()
    return [
        gpu_id
        for gpu_id, info in status.items()
        if not info["busy"] and (24000 - info["used_mb"]) >= min_free_mb
    ]


def should_terminate_early(exp: Experiment, baseline_curve: dict) -> Tuple[bool, str]:
    """根据 baseline 对比判断是否提前停止当前实验。

    第一、二阶段阈值更严格，第三阶段给足过渡步数后再判断。
    返回值为 (是否早停, 原因说明)。
    """
    if not baseline_curve:
        return False, ""

    t_env = exp.get_t_env()
    won = exp.get_battle_won()

    # 找到最接近的 baseline t_env
    baseline_t_envs = sorted(baseline_curve.keys())
    if not baseline_t_envs:
        return False, ""

    closest_t = min(baseline_t_envs, key=lambda x: abs(x - t_env))

    # 如果 t_env 差距太大（超过 100K），用最后可用的 baseline 值
    if abs(t_env - closest_t) > 100000 and baseline_t_envs:
        closest_t = baseline_t_envs[-1]

    baseline_at_t = baseline_curve.get(closest_t, 0)
    if baseline_at_t <= 0:
        return False, ""

    q_tot_steps = exp.config.get("q_tot_stage_steps", 500000)
    reward_steps = exp.config.get("reward_stage_steps", 0)
    stage2_end = q_tot_steps + reward_steps

    if t_env < stage2_end:
        # 第一、二阶段：低于同时间步 baseline 的 50%
        threshold = baseline_at_t * 0.5
        if won > 0 and won < threshold:
            return (
                True,
                f"Stage1/2 win={won:.3f} < {threshold:.3f} (50% of baseline@t={closest_t})",
            )
    else:
        # 第三阶段：连续 200K 步后仍低于同时间步 baseline 的 90%
        threshold = baseline_at_t * 0.9
        stage3_steps = t_env - stage2_end
        if won > 0 and won < threshold and stage3_steps > 200000:
            return (
                True,
                f"Stage3 win={won:.3f} < {threshold:.3f} (90% of baseline@t={closest_t}) for >200K",
            )

    return False, ""


def get_baseline_final_90(baseline_curve: dict) -> float:
    """计算 baseline 最终胜率中位数的 90% 阈值。"""
    if not baseline_curve:
        return 0.0
    sorted_items = sorted(baseline_curve.items())
    final_median = sorted_items[-1][1] if sorted_items else 0.0
    return final_median * 0.9


def build_command(exp: Experiment) -> str:
    """根据实验配置拼接启动命令。

    统一生成 `python3 -u src/main.py ... with ...` 形式的命令串。
    """
    base_cmd = "python3 -u src/main.py"

    # 核心配置
    config = exp.config.get("config", "qmix")
    env_config = exp.config.get("env_config", "sc2")
    cmd = f"{base_cmd} --config={config} --env-config={env_config}"

    # 标签
    if "tag" in exp.config:
        cmd += f" --tag={exp.tag}"

    # 收集 with 参数（env_args + overrides）
    with_args = []

    # 环境参数
    env_args = exp.config.get("env_args", {})
    for key, value in env_args.items():
        with_args.append(f"env_args.{key}={value}")

    # 覆盖参数
    overrides = exp.config.get("overrides", {})
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


def stream_process_output(
    process: subprocess.Popen, log_file: str
) -> Tuple[Optional[str], Optional[str]]:
    """实时转发子进程输出并提取关键元信息。

    在写日志的同时解析 Sacred run_id 与 TensorBoard 目录。
    """
    run_id_box = {"value": None}
    tb_dir_box = {"value": None}
    run_id_pattern = re.compile(r'Started run with ID "?(\d+)"?')
    tb_dir_pattern = re.compile(r"TensorBoard log dir:\s*(.+)$")

    def _reader():
        """后台线程：持续读取子进程输出并落盘。"""
        if process.stdout is None:
            return

        with open(log_file, "a", encoding="utf-8") as lf:
            for line in iter(process.stdout.readline, ""):
                lf.write(line)
                lf.flush()

                if run_id_box["value"] is None:
                    match = run_id_pattern.search(line)
                    if match:
                        run_id_box["value"] = match.group(1)

                if tb_dir_box["value"] is None:
                    tb_match = tb_dir_pattern.search(line)
                    if tb_match:
                        tb_dir_box["value"] = tb_match.group(1).strip()

            process.stdout.close()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # 最多等待 6 秒解析 Sacred run ID
    for _ in range(60):
        if run_id_box["value"] is not None:
            break
        if process.poll() is not None:
            break
        time.sleep(0.1)

    return run_id_box["value"], tb_dir_box["value"]


class ExperimentManager:
    """实验队列与 GPU 资源的调度器。"""

    def __init__(self, args):
        """初始化调度器状态、队列与 baseline 配置。"""
        self.args = args
        self.queue: List[dict] = []  # 实验配置队列
        self.running: Dict[int, Experiment] = {}  # GPU 编号到实验对象的映射
        self.completed: List[Experiment] = []
        self.notification_email = getattr(args, "notification_email", None)

        # 读取 baseline 曲线用于早停比较
        self.baseline_curve = load_baseline_curve(args.baseline_dir)
        self.baseline_final_90 = get_baseline_final_90(self.baseline_curve)
        print(f"Baseline final 90% threshold: {self.baseline_final_90:.3f}", flush=True)

        self.load_queue()

        os.makedirs("results/logs", exist_ok=True)

    def load_queue(self):
        """从一个或多个 YAML 文件加载实验队列。"""
        for config_file in self.args.exp_configs:
            path = Path(config_file)
            if not path.exists():
                print(f"WARNING: Config file not found: {config_file}", flush=True)
                continue

            with open(path) as f:
                data = yaml.safe_load(f)

            queue = data.get("queue", [])
            self.queue.extend(queue)

        print(f"Loaded {len(self.queue)} experiments into queue", flush=True)

    def launch_experiment(self, exp_config: dict, gpu_id: int) -> Experiment:
        """在指定 GPU 上启动单个实验并建立日志追踪。"""
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{gpu_id}"
        tag = exp_config.get("tag", "unnamed")

        exp = Experiment(exp_id=exp_id, tag=tag, config=exp_config, gpu_id=gpu_id)

        cmd = build_command(exp)
        log_file = f"results/logs/{exp_id}.log"

        # 设置 CUDA_VISIBLE_DEVICES 隔离到目标 GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"[LAUNCH] GPU {gpu_id} | {tag} | cmd: {cmd}", flush=True)

        with open(log_file, "w") as f:
            f.write(f"# GPU Manager launched experiment\n")
            f.write(f"# GPU: {gpu_id}\n")
            f.write(f"# Tag: {tag}\n")
            f.write(f"# Command: {cmd}\n")
            f.write(f"# Start time: {datetime.now()}\n")
            f.write("=" * 80 + "\n")

        process = subprocess.Popen(
            ["bash", "-c", cmd],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        sacred_run_id, tb_dir = stream_process_output(process, log_file)

        if sacred_run_id:
            print(f"[INFO] Sacred run ID: {sacred_run_id}", flush=True)
            # 保存真实 Sacred run_id，后续据此读取指标
            exp.exp_id = sacred_run_id

            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n# GPU Manager metadata\n")
                f.write(f"# sacred_run_id: {sacred_run_id}\n")
                f.write(f"# sacred_run_dir: results/sacred/{sacred_run_id}\n")
                if tb_dir:
                    f.write(f"# tensorboard_dir: {tb_dir}\n")
                else:
                    f.write("# tensorboard_dir: unknown\n")
                f.write("=" * 80 + "\n")
        else:
            # 回退方案：扫描最新 Sacred 目录
            print(
                f"[WARNING] Could not parse Sacred run ID, scanning for latest run...",
                flush=True,
            )
            sacred_dir = Path("results/sacred")
            if sacred_dir.exists():
                run_dirs = [d for d in sacred_dir.iterdir() if d.is_dir()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
                    exp.exp_id = latest_run.name
                    print(f"[INFO] Using latest Sacred run: {exp.exp_id}", flush=True)

                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n# GPU Manager metadata\n")
                        f.write(f"# sacred_run_id: {exp.exp_id}\n")
                        f.write(f"# sacred_run_dir: results/sacred/{exp.exp_id}\n")
                        if tb_dir:
                            f.write(f"# tensorboard_dir: {tb_dir}\n")
                        else:
                            f.write("# tensorboard_dir: unknown\n")
                        f.write("=" * 80 + "\n")
                else:
                    print(
                        f"[WARNING] No Sacred runs found, using exp_id: {exp_id}",
                        flush=True,
                    )
            else:
                print(
                    f"[WARNING] Sacred directory not found, using exp_id: {exp_id}",
                    flush=True,
                )

        exp.process = process
        exp.stage = "stage1"  # 初始阶段
        return exp

    def check_gpu_availability(self):
        """获取当前可用 GPU 列表。"""
        return get_free_gpus(min_free_mb=8000)

    def launch_if_possible(self):
        """当 GPU 空闲且队列非空时，尽可能启动更多实验。"""
        free_gpus = self.check_gpu_availability()

        # 排除已被占用的 GPU
        occupied_gpus = set(self.running.keys())
        available_gpus = [g for g in free_gpus if g not in occupied_gpus]

        # 按可用 GPU 数量尽可能多地启动实验
        while available_gpus and self.queue:
            gpu_id = available_gpus.pop(0)
            exp_config = self.queue.pop(0)

            exp = self.launch_experiment(exp_config, gpu_id)
            self.running[gpu_id] = exp
            print(
                f"[STATUS] Running: {len(self.running)}, Queued: {len(self.queue)}",
                flush=True,
            )

    def check_running_experiments(self):
        """检查运行中实验状态，并处理完成、崩溃与早停。"""
        for gpu_id, exp in list(self.running.items()):
            # 更新阶段
            prev_stage = exp.stage
            exp.stage = exp.determine_stage()

            # 进入第三阶段时记录第二阶段末尾胜率
            if prev_stage in ("stage1", "stage2") and exp.stage == "stage3":
                exp.stage2_end_win_rate = exp.get_battle_won()

            # 检查进程是否异常退出
            if exp.process and exp.process.poll() is not None:
                exp.stage = "crashed"
                self.completed.append(exp)
                del self.running[gpu_id]
                send_notification(
                    "Experiment Crashed",
                    f"{exp.tag} (GPU {gpu_id}) crashed unexpectedly. "
                    f"Check log: results/logs/{exp.exp_id}.log",
                    email_to=self.notification_email,
                )
                continue

            # 检查实验是否正常完成
            if exp.stage == "done":
                self.completed.append(exp)
                del self.running[gpu_id]
                print(f"[DONE] {exp.tag} completed on GPU {gpu_id}", flush=True)
                continue

            # 更新达到 baseline 90% 的最后时间步
            won = exp.get_battle_won()
            if won > 0 and self.baseline_final_90 > 0:
                if won >= self.baseline_final_90:
                    exp.last_time_above_baseline_90 = exp.get_t_env()

            # 检查是否需要早停
            should_stop, reason = should_terminate_early(exp, self.baseline_curve)
            if should_stop:
                print(f"[EARLY_STOP] {exp.tag} on GPU {gpu_id}: {reason}", flush=True)
                self.kill_experiment(exp)
                exp.kill_reason = reason
                self.completed.append(exp)
                del self.running[gpu_id]
                send_notification(
                    "Experiment Early Stopped",
                    f"{exp.tag} (GPU {gpu_id}) was stopped early. Reason: {reason}",
                    email_to=self.notification_email,
                )

    def kill_experiment(self, exp: Experiment):
        """终止指定实验进程。"""
        if exp.process and exp.process.poll() is None:
            exp.process.terminate()
            try:
                exp.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                exp.process.kill()
        exp.stage = "killed"
        print(
            f"[KILLED] {exp.tag} (PID {exp.process.pid if exp.process else 'N/A'})",
            flush=True,
        )

    def report_progress(self):
        """输出当前运行实验的进度摘要。"""
        if not self.running:
            return

        now = datetime.now()
        print(f"\n{'='*80}", flush=True)
        print(
            f"[{now.strftime('%Y-%m-%d %H:%M')}] {len(self.running)} experiments running:",
            flush=True,
        )
        for gpu_id, exp in sorted(self.running.items()):
            elapsed_min = (time.time() - exp.start_time) / 60
            t_env = exp.get_t_env()
            won = exp.get_battle_won()

            # 格式化显示 t_env
            if t_env >= 1000000:
                t_str = f"{t_env/1000000:.2f}M"
            elif t_env >= 1000:
                t_str = f"{t_env/1000:.0f}K"
            else:
                t_str = str(t_env)

            print(
                f"  GPU {gpu_id} | {exp.tag:20s} | {elapsed_min:5.1f}min | {exp.stage:7s} | "
                f"t_env={t_str:>8s} | won={won:.3f}",
                flush=True,
            )

        # 同时显示队列状态
        if self.queue:
            print(f"  Queue: {len(self.queue)} experiments waiting", flush=True)
        print(f"{'='*80}\n", flush=True)

    def cleanup_finished(self):
        """把已结束实验写入结果文件并清理缓存。"""
        for exp in self.completed:
            # 写入结果文件
            self.log_result(exp)

        # 清空完成列表
        self.completed.clear()

    def log_result(self, exp: Experiment):
        """将单个实验结果追加写入 results.tsv。"""
        results_file = Path("results.tsv")
        won = exp.get_battle_won()

        # 计算 stage3_drop_ratio
        if exp.stage2_end_win_rate > 0:
            stage3_drop_ratio = (
                exp.stage2_end_win_rate - won
            ) / exp.stage2_end_win_rate
        else:
            stage3_drop_ratio = None

        # 计算 sample_efficiency：最后一次达到 baseline 90% 的 t_env
        # 从未达到则记为 -1
        if exp.last_time_above_baseline_90 > 0:
            sample_efficiency = exp.last_time_above_baseline_90
        else:
            sample_efficiency = -1

        # 行格式：commit, 胜率, drop, sample_efficiency, 状态, 标签
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
        ).strip()

        drop_str = f"{stage3_drop_ratio:.3f}" if stage3_drop_ratio is not None else "na"
        eff_str = f"{sample_efficiency}" if sample_efficiency > 0 else "-1"

        row = f"{commit}\t{won:.6f}\t{drop_str}\t{eff_str}\t{exp.stage}\t{exp.tag}\n"

        with open(results_file, "a" if results_file.exists() else "w") as f:
            if not results_file.exists():
                f.write(
                    "commit\ttest_battle_won_mean\tstage3_drop_ratio\tsample_efficiency\tstatus\ttag\n"
                )
            f.write(row)

        print(
            f"[LOGGED] {exp.tag}: won={won:.6f}, drop={drop_str}, eff={eff_str}, status={exp.stage}",
            flush=True,
        )

    def run(self):
        """执行调度主循环，直到队列和运行任务全部清空。"""
        print("=" * 80, flush=True)
        print("GPU Manager starting...", flush=True)
        print(f"Queue size: {len(self.queue)}", flush=True)
        print(f"Check interval: {self.args.check_interval}s", flush=True)
        print("=" * 80, flush=True)

        try:
            # 首次尝试拉起实验
            self.launch_if_possible()

            while self.running or self.queue:
                time.sleep(self.args.check_interval)

                self.launch_if_possible()
                self.check_running_experiments()
                self.report_progress()
                self.cleanup_finished()

            print("All experiments completed. GPU Manager exiting.", flush=True)
        except Exception as e:
            send_notification(
                "GPU Manager Crashed",
                f"GPU Manager 异常退出。\n错误: {type(e).__name__}: {e}",
                email_to=self.notification_email,
            )
            raise


def main():
    """解析命令行参数并启动 ExperimentManager。"""
    parser = argparse.ArgumentParser(
        description="GPU Manager Orchestrator for PyMARL Autoresearch"
    )
    parser.add_argument(
        "--exp_configs",
        nargs="+",
        required=True,
        help="Experiment queue config YAML file(s)",
    )
    parser.add_argument(
        "--baseline_dir",
        default="results/baseline/MMM2",
        help="Path to QMIX baseline runs directory (default: results/baseline/MMM2)",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=1800,
        help="GPU check interval in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--notification_email",
        default=None,
        help="Email address to send crash notifications (requires SMTP_USERNAME/SMTP_PASSWORD in code)",
    )
    parser.add_argument(
        "--manager_log_file",
        default="gpu_manager.log",
        help="Manager output log file path (default: gpu_manager.log, set empty to disable)",
    )

    args = parser.parse_args()
    enable_manager_log(args.manager_log_file)

    try:
        manager = ExperimentManager(args)
        manager.run()
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        send_notification(
            "GPU Manager 启动失败",
            f"GPU Manager 在初始化时崩溃。\n错误: {type(e).__name__}: {e}",
            email_to=NOTIFICATION_EMAIL,
        )
        raise


if __name__ == "__main__":
    main()
