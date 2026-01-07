import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.rl_logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # 对参数进行健全性检查
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)  # 允许用“.”访问参数而非“[]”
    args.device = "cuda" if args.use_cuda else "cpu"

    # 初始化日志
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 配置日志中的 Tensorboard 相关内容
    unique_token = "{}__{}__{}__{}".format(
        args.tag,
        args.env_args["map_name"],
        args.name,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # 配置日志中的 Sacred
    logger.setup_sacred(_run)

    # ------------------------ 训练开始 ------------------------
    run_sequential(args=args, logger=logger)
    # ------------------------ 训练结束 ------------------------

    # 训练结束后处理所有子线程
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # 实例化 runner 以获得环境信息
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 从 env_info 中获取维度信息
    env_info = runner.get_env_info()
    """
    新增
    """
    args.env_info = env_info

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    """
    新增 (QPLEX)
    """
    args.unit_dim = env_info["unit_dim"]

    # 定义 scheme 和 groups
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # 实例化 buffer (基于 scheme 和 groups)
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # 实例化 controller (multiagent controller, mac)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 给 runner 设置 scheme、groups、preprocess 和 mac
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 实例化 learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    """
     * @author hyr
     * @modified 2025-11-25-20:48
     * @description 统一 Q 网络和 reward 网络加载模型的代码
    """
    if args.checkpoint_path != "":
        load_model(learner, args, logger, runner, args.checkpoint_path)

    if args.reward_checkpoint_path != "":
        load_model(
            learner,
            args,
            logger,
            runner,
            args.reward_checkpoint_path,
            is_reward_model=True,
        )

    # 开始训练
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    """
    新增
    训练 reward 网络
    """
    if args.two_stage_train and args.reward_checkpoint_path == "":
        reward_t_env = 0
        reward_episode = 0
        reward_last_log_T = 0
        reward_model_save_time = 0

        while reward_t_env <= args.reward_t_max:
            reward_t_env = learner.train_reward_network(
                None, reward_t_env, reward_episode
            )
            reward_episode += args.reward_batch_size

            # 打印日志
            if (reward_t_env - reward_last_log_T) >= args.log_interval:
                logger.log_stat("episode", reward_episode, reward_t_env)
                logger.print_recent_stats()
                last_log_T = reward_t_env

            """
             * @author hyr
             * @modified 2025-11-25-17:17
             * @description 保存 reward 模型
            """
            if args.save_reward_model and (
                reward_t_env - reward_model_save_time >= args.save_model_interval
                or reward_model_save_time == 0
            ):
                reward_model_save_time = reward_t_env
                reward_model_save_path = os.path.join(
                    args.local_results_path,
                    "reward_models",
                    args.unique_token,
                    str(reward_t_env),
                )
                os.makedirs(reward_model_save_path, exist_ok=True)
                logger.console_logger.info(
                    "Saving reward models to {}".format(reward_model_save_path)
                )
                learner.save_reward_models(reward_model_save_path)

    """
    训练 q 网络
    """
    while runner.t_env <= args.t_max:
        # 交互一个回合并把数据存入 buffer 中
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # 训练模型
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # 截断数据为当前 batch 中最长 episode 的长度
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            """
            修改
            learner 基于算法设计更新模型参数
            """
            if not args.two_stage_train and args.reward_mixer:
                learner.train_reward_network(episode_sample, runner.t_env, episode)
                learner.train_q_network(episode_sample, runner.t_env, episode)
            else:
                learner.train_q_network(episode_sample, runner.t_env, episode)

        # 测试模型
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )

            last_time = time.time()
            last_test_T = runner.t_env

            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 保存模型
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner 负责模型的保存/加载 —— 将 actor 的保存/加载委托给 mac
            learner.save_models(save_path)

        episode += args.batch_size_run

        # 打印日志
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


def load_model(learner, args, logger, runner, checkpoint_path, is_reward_model=False):
    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(checkpoint_path):
        logger.console_logger.info(
            "Checkpoint directiory {} doesn't exist".format(checkpoint_path)
        )
        return

    # Go through all files in checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if args.load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    model_path = os.path.join(checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    if not is_reward_model:
        learner.load_models(model_path)
        runner.t_env = timestep_to_load
    else:
        learner.load_reward_models(model_path)

    if args.evaluate or args.save_replay:
        evaluate_sequential(args, runner)
        return
