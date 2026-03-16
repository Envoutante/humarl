import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


plt.rcParams["font.family"] = ["DejaVu Sans Mono", "SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def smooth(scalars, weight):  # 指数平滑
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_npy_data(file_path, max_steps=2e6):
    """
    加载 .npy 文件中的数据

    参数:
    file_path: .npy 文件路径
    max_steps: 最大步数限制

    返回:
    steps: 步数列表
    values: 对应的值列表
    """
    try:
        data = np.load(file_path, allow_pickle=True)

        # 直接加载 win_rates_X.npy 格式的文件
        if "win_rates_" in file_path:
            # 这种格式的文件通常直接包含胜率值，没有步数信息
            # 我们需要生成步数信息
            values = data
            # 假设每个数据点间隔为 5000 步
            step_interval = 5000
            steps = np.arange(len(values)) * step_interval
            return steps, values

        # 假设 .npy 文件中的数据格式为 [steps, values] 或者是字典格式
        if isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[1] == 2:  # 如果是 [steps, values] 格式
                steps = data[:, 0]
                values = data[:, 1]
            else:
                # 如果是一维数组，假设它只包含值，我们需要生成步数
                if data.ndim == 1:
                    values = data
                    step_interval = 5000  # 假设每个数据点间隔为 10000 步
                    steps = np.arange(len(values)) * step_interval
                else:
                    print(f"Warning: Unsupported numpy array format in {file_path}")
                    return None, None
        elif isinstance(data, dict):  # 如果是字典格式
            if "steps" in data and "values" in data:
                steps = data["steps"]
                values = data["values"]
            else:
                # 尝试其他可能的键名
                possible_step_keys = ["step", "steps", "x", "iteration", "iterations"]
                possible_value_keys = [
                    "value",
                    "values",
                    "y",
                    "reward",
                    "rewards",
                    "return",
                    "returns",
                    "win_rate",
                    "win_rates",
                ]

                step_key = next((k for k in possible_step_keys if k in data), None)
                value_key = next((k for k in possible_value_keys if k in data), None)

                if step_key and value_key:
                    steps = data[step_key]
                    values = data[value_key]
                else:
                    print(f"Warning: Could not identify step/value keys in {file_path}")
                    return None, None
        else:
            print(f"Warning: Unsupported data type in {file_path}")
            return None, None

        # 限制步数
        if max_steps:
            mask = steps <= max_steps
            steps = steps[mask]
            values = values[mask]

        return steps, values
    except Exception as e:
        print(f"Error loading .npy file {file_path}: {e}")
        return None, None


def _format_step_label(step_value):
    if step_value >= 1e6:
        label = f"{step_value / 1e6:.1f}M"
    elif step_value >= 1e3:
        label = f"{step_value / 1e3:.0f}K"
    else:
        label = f"{int(step_value)}"
    return label


def plot_map_algorithms(
    map_name,
    scalar_name,
    smooth_weight,
    max_steps=2e6,
    phase_boundaries=None,
):
    # --- Robust Path Parsing ---
    path_parts = os.path.normpath(map_name).split(os.sep)
    if len(path_parts) >= 2:
        name_for_print, algo_name = path_parts[-2], path_parts[-1]
    else:
        # Fallback for simple names
        name_for_print, algo_name = map_name, map_name

    print(f"-------------{name_for_print} Result:-------")

    # The base_path is now the directory containing all runs for a single algorithm
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), map_name)

    # 颜色配置（扩展版）
    colors = [
        ["#017D6F", "#90EE90"],  # 绿色
        ["#6495ED", "#87CEFA"],  # 蓝色
        ["#CD5C5C", "#FA8072"],  # 红色
        ["#5E6C82", "#899FB0"],  # 灰色
        ["#af8fd0", "#caadd8"],  # 紫色
        ["#FF8C00", "#FFA07A"],  # 橙色
        ["#FFD700", "#FFFFE0"],  # 黄色
        ["#008B8B", "#E0FFFF"],  # 青色
        # ===== 新增配色 =====
        ["#8B4513", "#D2B48C"],  # 棕色（Earth tone，适合 baseline）
        ["#2F4F4F", "#B0C4DE"],  # 深蓝灰（稳重、对比强）
        ["#556B2F", "#C5E1A5"],  # 橄榄绿（低饱和，打印友好）
        ["#4B0082", "#B39DDB"],  # 靛蓝（区分紫色系）
        ["#A0522D", "#F5DEB3"],  # 土黄（接近自然色）
        ["#36454F", "#CFD8DC"],  # Charcoal（黑灰系，适合参考方法）
    ]

    plt.figure(figsize=(7, 6))
    # 设置网格线样式
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5

    # 目录结构（新）：地图名/算法/算法变体名/若干tb结果
    # 当传入 map_name="MMM2/QMIX" 时：base_path 指向 ".../MMM2/QMIX"，其下一层子目录为各个变体名。
    variant_names = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    variant_names.sort()

    plotted_any = False
    global_max_step = 0

    for idx, variant_name in enumerate(variant_names):
        variant_path = os.path.join(base_path, variant_name)

        all_steps = []
        all_values = []

        # 递归遍历该变体目录，收集所有 runs 的数据
        for root, _, files in os.walk(variant_path):
            for file_name in files:
                full_file_path = os.path.join(root, file_name)

                if file_name.endswith(".npy"):
                    steps, values = load_npy_data(full_file_path, max_steps=max_steps)
                    if steps is not None and values is not None and len(steps) > 0:
                        all_steps.append(steps)
                        all_values.append(values)
                    continue

                if file_name.startswith("events.out.tfevents"):
                    try:
                        ea = event_accumulator.EventAccumulator(full_file_path)
                        ea.Reload()

                        scalar_tags = ea.Tags()["scalars"]
                        target_scalar = None
                        if scalar_name in scalar_tags:
                            target_scalar = scalar_name
                        elif "eval_win_rate" in scalar_tags:
                            target_scalar = "eval_win_rate"
                        elif "incre_win_rate" in scalar_tags:
                            target_scalar = "incre_win_rate"

                        if target_scalar is None:
                            continue

                        scalar_data = ea.Scalars(target_scalar)
                        step_limit = max_steps + 5000 if max_steps else None
                        if step_limit is None:
                            steps = [s.step for s in scalar_data]
                            values = [s.value for s in scalar_data]
                        else:
                            steps = [
                                s.step for s in scalar_data if s.step <= step_limit
                            ]
                            values = [
                                s.value for s in scalar_data if s.step <= step_limit
                            ]
                        if steps:
                            all_steps.append(steps)
                            all_values.append(values)
                    except Exception as e:
                        print(f"Error processing {full_file_path}: {e}")

        if not all_values:
            print(
                f"Error: No data found for variant {variant_name} and scalar {scalar_name}"
            )
            continue

        # 找到最短的步数（按长度对齐）
        try:
            min_steps = min(len(steps) for steps in all_steps)
        except ValueError:
            print(
                f"Error: Could not determine minimum steps for variant {variant_name}"
            )
            continue

        aligned_values = [[] for _ in range(min_steps)]
        for values in all_values:
            if len(values) >= min_steps:
                for i in range(min_steps):
                    aligned_values[i].append(values[i])

        median_values = [np.median(v) for v in aligned_values]
        max_values = [np.max(v) for v in aligned_values]
        min_values = [np.min(v) for v in aligned_values]

        median_values = smooth(median_values, smooth_weight)
        max_values = smooth(max_values, smooth_weight)
        min_values = smooth(min_values, smooth_weight)

        # 统计：最后250K步窗口内，中位数曲线的最大值 ± 每个run峰值的std
        if all_steps and all_steps[0]:
            step_interval = (
                all_steps[0][1] - all_steps[0][0] if len(all_steps[0]) > 1 else 5000
            )
            steps_per_250k = int(250000 / step_interval) if step_interval > 0 else 0
            if len(median_values) > steps_per_250k > 0:
                last_250k_median = median_values[-steps_per_250k:]
                max_median_250k = np.max(last_250k_median)

                peak_values_per_run = []
                for v_list in all_values:
                    if len(v_list) > steps_per_250k:
                        last_window = v_list[-steps_per_250k:]
                        peak_values_per_run.append(np.max(last_window))

                final_std_report = (
                    np.std(peak_values_per_run) if peak_values_per_run else 0.0
                )
                print(
                    f"{variant_name} 统计结果: {max_median_250k:.4f} ± {final_std_report:.4f}"
                )

        plt.plot(
            all_steps[0][:min_steps],
            median_values,
            color=colors[idx % len(colors)][0],
            label=variant_name,
            linewidth=3,
        )
        plt.fill_between(
            all_steps[0][:min_steps],
            min_values,
            max_values,
            alpha=0.6,
            color=colors[idx % len(colors)][1],
        )
        if min_steps > 0:
            global_max_step = max(global_max_step, all_steps[0][min_steps - 1])
        plotted_any = True

    if not plotted_any:
        print(f"Error: No variants plotted under {base_path}")
        return

    plt.xlabel("训练时间步", fontsize=20)
    plt.ylabel("测试胜率", fontsize=20)
    plt.title(f"{algo_name}在{name_for_print}的测试胜率", fontsize=20)

    # 根据阶段边界画虚线，并将边界加入 x 轴刻度
    if phase_boundaries is not None:
        ax = plt.gca()
        x_axis_max = max_steps if max_steps else global_max_step
        filtered_boundaries = sorted(b for b in phase_boundaries if 0 < b < x_axis_max)

        for boundary in filtered_boundaries:
            plt.axvline(
                x=boundary,
                linestyle="--",
                color="#444444",
                linewidth=1.4,
                alpha=0.9,
            )

        # 保留原有自动刻度（对应原网格），仅额外插入阶段边界相关刻度
        original_ticks = [t for t in ax.get_xticks() if 0 <= t <= x_axis_max]
        tick_values = original_ticks + [0] + filtered_boundaries + [x_axis_max]
        tick_values = sorted(set(float(v) for v in tick_values))
        tick_labels = [_format_step_label(v) for v in tick_values]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)

    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True)

    # --- Save Plot to Correct Directory ---
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Generate a filename for the plot
    output_filename = f"{name_for_print}_{algo_name}_win_rate.png"
    # Create the full path to save the file in the script's directory
    save_path = os.path.join(script_dir, output_filename)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Disabled for SSH compatibility


if __name__ == "__main__":
    # 直接分析QMIX文件夹下的所有算法
    # # SMACv2
    # plot_map_algorithms('gen_zerg/GATMIX', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('gen_protoss/GATMIX', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('gen_terran/GATMIX', 'test_battle_won_mean', 0.8)

    # plot_map_algorithms('gen_protoss/ADD_O_Test', 'test_battle_won_mean', 0.8)

    # # SMAC
    # plot_map_algorithms('6h_vs_8z/GATMIX-Test4', 'test_battle_won_mean', 0.8)
    plot_map_algorithms(
        "3s5z/res_reg_multi",
        "test_battle_won_mean",
        0.8,
        max_steps=2.6e6,
        phase_boundaries=[0.5e6, 0.8e6],
    )
    # plot_map_algorithms('2s3z_vs_2s4z/GATMIX', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('5m_vs_6m/GATMIX', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('MMM2/GATMIX', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('corridor/GATMIX', 'test_battle_won_mean', 0.8)

    # plot_map_algorithms('6h_vs_8z/ablation', 'test_battle_won_mean', 0.8)
    # plot_map_algorithms('3s5z_vs_3s6z/ablation', 'test_battle_won_mean', 0.8)

    # plot_map_algorithms('Ablation/ALL_SUM', 'test_battle_won_mean', 0.8)
