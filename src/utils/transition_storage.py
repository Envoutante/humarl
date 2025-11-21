import h5py
import numpy as np
import torch as th
import json

class TransitionStorage:
    def __init__(self, args, filename, max_size=50000):
        self.args = args
        self.filename = filename
        self.max_size = max_size
        self.current_index = 0
        self.is_full = False
        
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
        
        self._init_storage()
    
    def _init_storage(self):
        """
        初始化 HDF5 文件的结构
        """
        with h5py.File(self.filename, 'a') as f:
            """
            对于旧文件
            """
            # 根据元数据设置索引
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
                self.current_index = metadata['current_index']
                self.is_full = metadata['is_full']
                return
            
            """
            对于新文件
            """
            # 创建 dataset
            for key, info in self.scheme.items():
                vshape = info["vshape"]
                dtype = info.get("dtype", np.float32)
                
                # 设置分组数据的维度
                if "group" in info:
                    group_size = self.groups[info["group"]]
                    actual_shape = (self.max_size, self.args.env_info["episode_limit"] + 1, group_size, *vshape)
                # 设置非分组数据的维度
                else:
                    actual_shape = (self.max_size, self.args.env_info["episode_limit"] + 1, *vshape)
                
                if key not in f:
                    f.create_dataset(key, actual_shape, dtype=dtype)
            
            # 保存元数据
            metadata = {
                'max_size': self.max_size,
                'current_index': 0,
                'is_full': False
            }
            f.attrs['metadata'] = json.dumps(metadata)


    def save_transition_batch(self, batch_data):
        """
        保存一个 batch 的 transition
        """
        with h5py.File(self.filename, 'a') as f:
            batch_size = None
            
            # 确定 batch_size
            for key in self.scheme.keys():
                if key in batch_data.data.transition_data:
                    batch_size = batch_data[key].shape[0]  # [batch_size, ...]
                break
            
            if batch_size is None:
                raise ValueError("No valid data found in batch_data")
            
            # 计算存储位置
            start_idx = self.current_index
            end_idx = self.current_index + batch_size
            
            if end_idx > self.max_size:
                # 循环覆盖
                end_idx = self.max_size
                self.is_full = True
                batch_size = self.max_size - self.current_index
            
            # 保存每个字段的数据
            for key in self.scheme.keys():
                if key in batch_data.data.transition_data:
                    data = batch_data[key]
                    
                    # 确保数据在 CPU 上并且是 numpy 数组
                    if th.is_tensor(data):
                        data = data.cpu().numpy()
                    
                    # 截取实际可存储的数据量
                    actual_data = data[:batch_size]
                    
                    # 保存到 HDF5 数据集
                    f[key][start_idx:end_idx] = actual_data
            
            # 更新索引
            self.current_index = end_idx % self.max_size
            
            # 更新元数据
            metadata = json.loads(f.attrs['metadata'])
            metadata['current_index'] = self.current_index
            metadata['is_full'] = self.is_full
            metadata['total_stored'] = end_idx if not self.is_full else self.max_size
            f.attrs['metadata'] = json.dumps(metadata)


    def load_transition_batch(self, batch_size=32):
        """
        加载一个 batch 的 transition
        """
        with h5py.File(self.filename, 'r') as f: 
            batch_data = {}
            for key in self.scheme.keys():
                if key in f:
                    batch_data[key] = f[key][:batch_size]
            
            return batch_data


if __name__ == "__main__":
    file_path = "results/collected_transitions/transitions_hyr__3s5z__qmix__2025-11-21_19-05-35.h5"
    
    with h5py.File(file_path, 'r') as f:
        for attr_name, attr_value in f.attrs.items():
            print(f"{attr_name}: {attr_value}")

        for key in list(f.keys())[:]:
            data = f[key]
            print(f"{key}: {data.shape}")

