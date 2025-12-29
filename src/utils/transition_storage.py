import h5py
import numpy as np
import torch as th
import json
import time
import pandas as pd
import os


class TransitionStorage:
    def __init__(self, args, filename, max_size=50000):
        self.args = args
        self.filename = filename
        self.max_size = max_size
        self.current_index = 0
        self.is_full = False

        # 提取 episode_limit
        if not th.is_tensor(self.args.env_info["episode_limit"]):
            self.episode_limit = self.args.env_info["episode_limit"]
        else:
            self.episode_limit = self.args.env_info["episode_limit"].item()

        self.scheme = {
            "state": {"vshape": (self.args.env_info["state_shape"],)},
            "obs": {"vshape": (self.args.env_info["obs_shape"],), "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": np.int64},
            "avail_actions": {
                "vshape": (self.args.env_info["n_actions"],),
                "group": "agents",
                "dtype": np.int32,
            },
            "actions_onehot": {
                "vshape": (self.args.env_info["n_actions"],),
                "group": "agents",
                "dtype": np.int32,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": np.uint8},
            "filled": {"vshape": (1,), "dtype": np.int64},
        }
        self.groups = {"agents": self.args.env_info["n_agents"]}

        self._init_storage()

    def _init_storage(self):
        """
        初始化 HDF5 文件的结构
        """
        with h5py.File(self.filename, "a") as f:
            # 如果文件已存在，加载元数据
            if "metadata" in f.attrs:
                metadata = json.loads(f.attrs["metadata"])
                self.current_index = metadata.get("current_index", 0)
                self.is_full = metadata.get("is_full", False)
                return

            # 新文件: 初始化元数据
            # 每个 episode 的 group 和 dataset 将在 save_transition_batch 时动态创建
            metadata = {
                "max_size": self.max_size,
                "current_index": 0,
                "is_full": False,
                "total_episodes": 0,
            }
            f.attrs["metadata"] = json.dumps(metadata)

    def save_transition_batch(self, batch_data):
        """
        写入数据到 HDF5 文件
        为每个 episode 创建一个 group, group 内包含所有字段的 dataset
        支持传入包含多个 episode 的 batch
        """
        with h5py.File(self.filename, "a") as f:
            # 确定 batch_size（episode 数）
            batch_size = None
            for key in self.scheme.keys():
                if (
                    hasattr(batch_data.data, "transition_data")
                    and key in batch_data.data.transition_data
                ):
                    batch_size = batch_data[key].shape[0]
                    break
                elif isinstance(batch_data, dict) and key in batch_data:
                    batch_size = batch_data[key].shape[0]
                    break

            if batch_size is None:
                raise ValueError("No valid data found in batch_data")

            start_idx = self.current_index
            actual_saved = 0

            # 为每个 episode 创建 group 并存储所有字段
            for ep_idx in range(batch_size):
                episode_idx = start_idx + ep_idx
                
                # 检查是否超出最大容量
                if episode_idx >= self.max_size:
                    self.is_full = True
                    break
                
                # 创建或获取 episode group
                group_name = f"episode_{episode_idx}"
                if group_name not in f:
                    ep_group = f.create_group(group_name)
                else:
                    ep_group = f[group_name]

                # 为当前 episode 存储所有字段
                for key, info in self.scheme.items():
                    # 获取数据
                    if (
                        hasattr(batch_data.data, "transition_data")
                        and key in batch_data.data.transition_data
                    ):
                        data = batch_data[key]
                    elif isinstance(batch_data, dict) and key in batch_data:
                        data = batch_data[key]
                    else:
                        continue

                    if th.is_tensor(data):
                        data = data.cpu().numpy()

                    # 提取当前 episode 的数据
                    episode_data = data[ep_idx]  # shape: (seq_len, ...)
                    
                    # 确定 dataset 的形状和类型
                    vshape = info["vshape"]
                    dtype = info.get("dtype", np.float32)
                    
                    if "group" in info:
                        group_size = self.groups[info["group"]]
                        dataset_shape = (self.episode_limit + 1, group_size, *vshape)
                    else:
                        dataset_shape = (self.episode_limit + 1, *vshape)

                    # 创建或更新 dataset
                    if key not in ep_group:
                        ep_group.create_dataset(
                            key,
                            dataset_shape,
                            dtype=dtype,
                            data=episode_data,
                            compression="gzip",
                            compression_opts=4,
                        )
                    else:
                        # 如果 dataset 已存在，直接写入
                        ep_group[key][:] = episode_data
                
                actual_saved += 1

            # 更新指针
            self.current_index = start_idx + actual_saved
            if self.current_index >= self.max_size:
                self.is_full = True
                self.current_index = self.max_size

            # 更新元数据
            metadata = json.loads(f.attrs["metadata"])
            metadata["current_index"] = self.current_index
            metadata["is_full"] = self.is_full
            metadata["total_episodes"] = (
                self.max_size if self.is_full else self.current_index
            )
            f.attrs["metadata"] = json.dumps(metadata)

    def load_transition_batch(self, batch_size=32):
        """
        从 HDF5 文件读出数据
        按 episode 抽样读取一个 batch
        """
        with h5py.File(self.filename, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
            total_episodes = metadata.get("total_episodes", 0)

            if total_episodes == 0:
                raise ValueError("No episode stored in the buffer.")

            random_indices = np.random.choice(
                total_episodes, size=batch_size, replace=True
            )

            batch_data = {}
            start_time = time.time()

            # 从 episode groups 读取
            for key in self.scheme.keys():
                episode_list = []
                for ep_idx in random_indices:
                    group_name = f"episode_{ep_idx}"
                    if group_name in f and key in f[group_name]:
                        episode_data = np.array(f[group_name][key])
                        episode_list.append(episode_data)
                    else:
                        raise ValueError(
                            f"Episode {ep_idx} or field {key} not found in storage."
                        )
                batch_data[key] = np.array(episode_list)

            end_time = time.time()
            print(f"Episode-based 采样耗时: {(end_time - start_time):.4f} 秒")

            return batch_data


def _flatten_to_dataframe(data, key):
    """
    将多维数组展平为 DataFrame
    data shape: (batch_size, seq_len, ...)
    输出: DataFrame, 每行代表一个 episode, 每列代表一个 step_idx
    """
    batch_size, seq_len = data.shape[0], data.shape[1]

    # 为每个 episode 创建一行
    rows = []
    for ep_idx in range(batch_size):
        row = {"episode_idx": ep_idx}

        # 遍历每个 step, 将数据展平后添加到行中
        for step_idx in range(seq_len):
            step_data = data[ep_idx, step_idx]

            # 根据维度处理
            if step_data.ndim == 0:
                # 标量
                col_name = f"step_{step_idx}"
                row[col_name] = step_data.item()
            elif step_data.ndim == 1:
                # 一维数组, 展平为多列
                for i, val in enumerate(step_data):
                    if len(step_data) == 1:
                        col_name = f"step_{step_idx}"
                    else:
                        col_name = f"step_{step_idx}_feat{i}"
                    row[col_name] = val
            elif step_data.ndim == 2:
                # 二维数组 (如 n_agents x feature_dim), 展平
                for i in range(step_data.shape[0]):
                    for j in range(step_data.shape[1]):
                        col_name = f"step_{step_idx}_agent{i}_feat{j}"
                        row[col_name] = step_data[i, j]
            else:
                # 更高维度, 完全展平
                flat = step_data.flatten()
                for i, val in enumerate(flat):
                    col_name = f"step_{step_idx}_flat{i}"
                    row[col_name] = val

        rows.append(row)

    return pd.DataFrame(rows)


def _combine_dataframes(batch_data_dict):
    """
    将所有字段合并到一个 DataFrame
    """
    batch_size = None
    seq_len = None
    all_keys = list(batch_data_dict.keys())

    # 确定 batch_size 和 seq_len
    for key, data in batch_data_dict.items():
        if batch_size is None:
            batch_size, seq_len = data.shape[0], data.shape[1]
        else:
            assert (
                data.shape[0] == batch_size and data.shape[1] == seq_len
            ), f"所有字段必须有相同的 (batch_size, seq_len) 维度"

    # 构建合并后的数据
    combined_rows = []

    for ep_idx in range(batch_size):
        # 为当前 episode 的每个字段创建一行
        for field_idx, key in enumerate(all_keys):
            data = batch_data_dict[key]
            row = {
                "episode_idx": ep_idx,
                "field_name": key,
                "field_idx": field_idx,
            }

            # 遍历每个 step, 将数据展平后添加到行中
            for step_idx in range(seq_len):
                step_data = data[ep_idx, step_idx]

                # 根据维度处理
                if step_data.ndim == 0:
                    # 标量
                    col_name = f"step_{step_idx}"
                    row[col_name] = step_data.item()
                elif step_data.ndim == 1:
                    # 一维数组, 展平为多列
                    for i, val in enumerate(step_data):
                        if len(step_data) == 1:
                            col_name = f"step_{step_idx}"
                        else:
                            col_name = f"step_{step_idx}_feat{i}"
                        row[col_name] = val
                elif step_data.ndim == 2:
                    # 二维数组 (如 n_agents x feature_dim), 展平
                    for i in range(step_data.shape[0]):
                        for j in range(step_data.shape[1]):
                            col_name = f"step_{step_idx}_agent{i}_feat{j}"
                            row[col_name] = step_data[i, j]
                else:
                    # 更高维度, 完全展平
                    flat = step_data.flatten()
                    for i, val in enumerate(flat):
                        col_name = f"step_{step_idx}_flat{i}"
                        row[col_name] = val

            combined_rows.append(row)

    return pd.DataFrame(combined_rows)


if __name__ == "__main__":
    file_path = "results/collected_transitions/new.h5"
    batch_size = 32

    with h5py.File(file_path, "r") as f:
        """
        查看文件元数据
        """
        for attr_name, attr_value in f.attrs.items():
            print(f"{attr_name}: {attr_value}")

        # 获取元数据
        metadata = json.loads(f.attrs["metadata"])
        
        """
        查看各数据集的形状
        """
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        print(f"\n找到 {len(episode_keys)} 个 episode groups")
        if episode_keys:
            # 显示第一个 episode 的结构
            first_ep = f[episode_keys[0]]
            print(f"\n第一个 episode ({episode_keys[0]}) 包含的字段:")
            for key in first_ep.keys():
                print(f"  {key}: {first_ep[key].shape}")

        """
        查看任意 batch_size 大小的数据, 并保存为 CSV
        """
        # 获取 total_stored
        total_episodes = metadata.get("total_episodes", 0)
        # 生成随机索引
        random_indices = np.random.choice(
            total_episodes, size=batch_size, replace=False
        )

        # 只收集 reward、terminated、filled 这三个字段
        target_fields = ["reward", "terminated", "filled"]
        batch_data_dict = {}
        
        # 记录采样耗时
        sample_start_time = time.time()
        
        # 从 episode groups 读取
        for key in target_fields:
            episode_list = []
            for ep_idx in random_indices:
                group_name = f"episode_{ep_idx}"
                if group_name in f and key in f[group_name]:
                    episode_data = np.array(f[group_name][key])
                    episode_list.append(episode_data)
                else:
                    print(f"警告: Episode {ep_idx} 或字段 {key} 不存在")
                    break
            if episode_list:
                batch_data = np.array(episode_list)
                batch_data_dict[key] = batch_data
                print(f"{key}: shape={batch_data.shape}")
        
        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time
        print(f"\n采样 {batch_size} 个 episode 耗时: {sample_duration:.4f} 秒")

        if not batch_data_dict:
            print("错误: 没有找到任何目标字段, 无法保存 CSV")
            exit(1)

        # 保存为 CSV
        csv_output_dir = "results/collected_transitions/csv_exports"
        os.makedirs(csv_output_dir, exist_ok=True)

        # 方法1: 将每个字段保存为单独的 CSV 文件（每行一个episode, 每列一个step）
        for key, data in batch_data_dict.items():
            df = _flatten_to_dataframe(data, key)
            csv_path = os.path.join(csv_output_dir, f"{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"已保存 {key} 到 {csv_path}")

        # 方法2: 将所有字段合并到一个 CSV（每个episode的3个字段按行排列）
        try:
            combined_df = _combine_dataframes(batch_data_dict)
            combined_csv_path = os.path.join(csv_output_dir, "combined_transitions.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"已保存合并数据到 {combined_csv_path}")
        except Exception as e:
            print(f"无法合并所有字段到一个 CSV: {e}")
            print("已为每个字段单独保存 CSV 文件")
