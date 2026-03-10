from collections import defaultdict
import logging
import numpy as np
import torch


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.tb_mode = None
        self._tb_group_layout_keys = set()

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Prefer SummaryWriter; avoid add_scalars to prevent creating extra TB runs.
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=directory_name)
            self.tb_mode = "summary_writer"
        except Exception:
            # Fallback for environments relying on tensorboard_logger.
            from tensorboard_logger import configure, log_value

            configure(directory_name)
            self.tb_logger = log_value
            self.tb_mode = "tensorboard_logger"
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            if getattr(self, "tb_mode", None) == "summary_writer":
                self.tb_writer.add_scalar(key, value, t)
            else:
                self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def log_stat_group(self, key, value_dict, t, to_sacred=True):
        """Log grouped scalar curves (same chart) with labels from value_dict keys."""
        if not value_dict:
            return

        # Keep the original main key for "true" so old dashboards remain stable.
        series = []
        for label, value in value_dict.items():
            if label == "true":
                sub_key = key
            else:
                sub_key = "{}/{}".format(key, label)
            series.append((sub_key, value))

        for sub_key, value in series:
            self.log_stat(sub_key, value, t, to_sacred=to_sacred)

        # Add a grouped custom chart without creating extra runs.
        if (
            self.use_tb
            and getattr(self, "tb_mode", None) == "summary_writer"
            and key not in self._tb_group_layout_keys
            and len(series) > 1
        ):
            try:
                layout = {
                    "Grouped": {key: ["Multiline", [sub_key for sub_key, _ in series]]}
                }
                self.tb_writer.add_custom_scalars(layout)
                self._tb_group_layout_keys.add(key)
            except Exception:
                # If custom layout is unavailable, scalar tags are still logged normally.
                pass

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            # item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            values = torch.tensor(
                [x[1] for x in self.stats[k][-window:]], dtype=torch.float32
            )
            item = "{:.4f}".format(values.mean().item())

            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
