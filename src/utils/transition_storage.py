import h5py
import numpy as np
import torch as th
import json
import time

class TransitionStorage:
    def __init__(self, args, filename, max_size=50000):
        self.args = args
        self.filename = filename
        self.max_size = max_size
        self.current_index = 0
        self.is_full = False
        
        # 提取 episode_limit，用于内存预分配
        if not th.is_tensor(self.args.env_info["episode_limit"]):
            self.episode_limit = self.args.env_info["episode_limit"]
        else:
            self.episode_limit = self.args.env_info["episode_limit"].item()
            
        self.scheme = {
            "state": {"vshape": (self.args.env_info["state_shape"],)},
            "obs": {"vshape": (self.args.env_info["obs_shape"],), "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": np.int64},
            "avail_actions": {"vshape": (self.args.env_info["n_actions"],), "group": "agents", "dtype": np.int32},
            "actions_onehot": {"vshape": (self.args.env_info["n_actions"],), "group": "agents", "dtype": np.int32},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": np.uint8},
            "filled": {"vshape": (1,), "dtype": np.int64},
        }
        self.groups = {
            "agents": self.args.env_info["n_agents"]
        }
        
        # 优化点：初始化 HDF5 文件结构并加载数据到内存
        self.in_memory_data = {}
        self._init_storage()
        self._load_to_memory() 
    
    def _init_storage(self):
        """
        初始化 HDF5 文件的结构，如果文件存在则读取元数据。
        """
        with h5py.File(self.filename, 'a') as f:
            
            # --- 对于旧文件：读取元数据 ---
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
                self.current_index = metadata.get('current_index', 0)
                self.is_full = metadata.get('is_full', False)
                return
            
            # --- 对于新文件：创建 dataset 并写入元数据 ---
            for key, info in self.scheme.items():
                vshape = info["vshape"]
                dtype = info.get("dtype", np.float32)
                
                # 设置维度：[max_size, episode_limit + 1, ...]
                if "group" in info:
                    group_size = self.groups[info["group"]]
                    actual_shape = (self.max_size, self.episode_limit + 1, group_size, *vshape)
                else:
                    actual_shape = (self.max_size, self.episode_limit + 1, *vshape)
                
                if key not in f:
                    # 使用 gzip 压缩，减少磁盘占用 (可选，但推荐)
                    f.create_dataset(key, actual_shape, dtype=dtype, chunks=True, compression='gzip') 
            
            # 保存元数据
            metadata = {
                'max_size': self.max_size,
                'current_index': 0,
                'is_full': False,
                'total_episodes': 0 
            }
            f.attrs['metadata'] = json.dumps(metadata)

    def _load_to_memory(self):
        """
        优化点：将 HDF5 中的所有数据加载到内存中的字典，用于快速采样。
        """
        # 预分配内存空间，即使当前 HDF5 中没有数据
        for key, info in self.scheme.items():
            vshape = info["vshape"]
            dtype = info.get("dtype", np.float32)
            
            if "group" in info:
                group_size = self.groups[info["group"]]
                actual_shape = (self.max_size, self.episode_limit + 1, group_size, *vshape)
            else:
                actual_shape = (self.max_size, self.episode_limit + 1, *vshape)
            
            # 预分配一个完整的 NumPy 数组
            self.in_memory_data[key] = np.zeros(actual_shape, dtype=dtype)
        
        # 从 HDF5 文件加载已有的数据
        with h5py.File(self.filename, 'r') as f:
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
                total_episodes = metadata.get('total_episodes', 0)
                
                if total_episodes > 0:
                    # HDF5 文件中有数据，需要处理环形 buffer 的读取逻辑
                    current_index = metadata['current_index']
                    is_full = metadata['is_full']
                    
                    for key in self.scheme.keys():
                        if key in f:
                            dataset = f[key]
                            
                            if is_full:
                                # 1. 读出从 current_index 到 max_size 的部分 (旧数据)
                                idx_start_old = current_index
                                idx_end_old = self.max_size
                                data_old = np.array(dataset[idx_start_old:idx_end_old])
                                
                                # 2. 读出从 0 到 current_index 的部分 (新数据)
                                idx_start_new = 0
                                idx_end_new = current_index
                                data_new = np.array(dataset[idx_start_new:idx_end_new])
                                
                                # 3. 重新拼接成逻辑上的连续数组
                                # 拼接顺序：[新数据段] + [旧数据段]
                                self.in_memory_data[key][:current_index] = data_new
                                self.in_memory_data[key][current_index:] = data_old
                                
                            else:
                                # 直接读出从 0 到 current_index 的部分
                                self.in_memory_data[key][:current_index] = np.array(dataset[:current_index])
    
    def save_transition_batch(self, batch_data):
        """
        同时写入 HDF5 文件和内存中的 Replay Buffer。
        """
        with h5py.File(self.filename, 'a') as f:
            batch_size = None

            # 确定 batch_size（episode 数）
            for key in self.scheme.keys():
                # 假设 batch_data 是一个结构，包含 transition_data 属性
                if hasattr(batch_data.data, 'transition_data') and key in batch_data.data.transition_data:
                    batch_size = batch_data[key].shape[0]  # [batch, ep_len, ...]
                    break
                # 如果 batch_data 直接是字典或类似结构
                elif isinstance(batch_data, dict) and key in batch_data:
                    batch_size = batch_data[key].shape[0]
                    break
                    
            if batch_size is None:
                raise ValueError("No valid data found in batch_data")

            start_idx = self.current_index
            end_idx = self.current_index + batch_size

            is_wrapping = False # 是否发生循环覆盖
            if end_idx > self.max_size:
                is_wrapping = True
                self.is_full = True
                
                # 截断到剩余空间
                write_size_part1 = self.max_size - self.current_index
                write_size_part2 = batch_size - write_size_part1
                
                # 写入到末尾
                for key in self.scheme.keys():
                    if hasattr(batch_data.data, 'transition_data') and key in batch_data.data.transition_data:
                        data = batch_data[key]
                    elif isinstance(batch_data, dict) and key in batch_data:
                        data = batch_data[key]
                    else:
                        continue
                        
                    if th.is_tensor(data):
                        data = data.cpu().numpy()

                    # Part 1: 写入到 HDF5 的末尾
                    actual_data_part1 = data[:write_size_part1]
                    f[key][start_idx:self.max_size] = actual_data_part1
                    
                    # 写入到内存的末尾
                    self.in_memory_data[key][start_idx:self.max_size] = actual_data_part1

                    # Part 2: 从 HDF5 的开头写入 (覆盖)
                    actual_data_part2 = data[write_size_part1:]
                    f[key][0:write_size_part2] = actual_data_part2
                    
                    # 写入到内存的开头
                    self.in_memory_data[key][0:write_size_part2] = actual_data_part2
                
                # 更新指针：回到开头
                self.current_index = write_size_part2
                
            else:
                # 连续写入，未发生循环覆盖
                for key in self.scheme.keys():
                    if hasattr(batch_data.data, 'transition_data') and key in batch_data.data.transition_data:
                        data = batch_data[key]
                    elif isinstance(batch_data, dict) and key in batch_data:
                        data = batch_data[key]
                    else:
                        continue
                        
                    if th.is_tensor(data):
                        data = data.cpu().numpy()

                    actual_data = data[:batch_size]
                    
                    # 写入 HDF5
                    f[key][start_idx:end_idx] = actual_data
                    
                    # 写入内存
                    self.in_memory_data[key][start_idx:end_idx] = actual_data
                    
                # 更新指针
                self.current_index = end_idx

            # 更新元数据
            metadata = json.loads(f.attrs['metadata'])
            metadata['current_index'] = self.current_index
            metadata['is_full'] = self.is_full
            metadata['total_episodes'] = self.max_size if self.is_full else self.current_index
            f.attrs['metadata'] = json.dumps(metadata)


    def load_transition_batch(self, batch_size=32):
        """
        优化点：完全从内存中采样读取，无需 HDF5 I/O，速度极快。
        """
        # 计算已存储的 episode 总数
        total_episodes = self.max_size if self.is_full else self.current_index

        if total_episodes == 0:
            raise ValueError("No episode stored in the buffer.")

        # 随机选择索引
        random_indices = np.random.choice(total_episodes, size=batch_size, replace=True)

        batch_data = {}
        start_time = time.time()
        
        # 批量切片：只进行内存操作
        for key in self.scheme.keys():
            if key in self.in_memory_data:
                # 内存切片速度比 HDF5 I/O 快得多
                batch_data[key] = self.in_memory_data[key][random_indices]
                
        end_time = time.time()
        print(f"内存采样耗时: {(end_time - start_time):.4f} 秒")

        return batch_data


if __name__ == "__main__":
    file_path = "results/collected_transitions/transitions_1.h5"
    batch_size = 32
    
    with h5py.File(file_path, 'r') as f:
        """
        查看文件元数据
        """
        for attr_name, attr_value in f.attrs.items():
            print(f"{attr_name}: {attr_value}")

        """
        查看各数据集的形状
        """
        for key in list(f.keys())[:]:
            data = f[key]
            print(f"{key}: {data.shape}")

        """
        查看任意 batch_size 大小的数据
        """
        # 获取 total_stored
        metadata = json.loads(f.attrs['metadata'])
        total_stored = metadata.get('total_stored', 0)
        # 生成随机索引
        random_indices = np.random.choice(total_stored, size=batch_size, replace=False)
        for key in list(f.keys())[:]:
            data = np.array(f[key])
            batch_data = data[random_indices]
