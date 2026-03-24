import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


plt.rcParams["font.family"] = ["DejaVu Sans Mono", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def smooth(scalars, weight):
    if not scalars:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def _event_files(root_dir):
    event_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file_name))
    event_files.sort()
    return event_files


def _normalize_base_tag(tag):
    lower = tag.lower()
    lower = re.sub(r"[/.\\-]+", "_", lower)
    # Strip true/pred marker words so both tags can be matched to same base.
    lower = re.sub(
        r"(?:^|_)"
        r"(true|pred|prediction|target|gt|groundtruth|ground_truth)"
        r"(?:_|$)",
        "_",
        lower,
    )
    lower = re.sub(r"_+", "_", lower).strip("_")
    return lower


def _detect_true_pred_pair(scalar_tags):
    return_tags = [t for t in scalar_tags if "return" in t.lower()]
    if not return_tags:
        return None, None

    # Priority 1: match "base" and "base/pred" (or "base_pred") pairs,
    # e.g. test_return_mean and test_return_mean/pred.
    lower_to_original = {t.lower(): t for t in return_tags}
    pair_candidates = []
    for tag in return_tags:
        lower = tag.lower()
        match = re.match(r"^(.*?)(?:[/_.-]pred(?:iction)?)$", lower)
        if not match:
            continue

        base_lower = match.group(1).strip()
        if not base_lower:
            continue

        if base_lower in lower_to_original:
            true_tag = lower_to_original[base_lower]
            pred_tag = tag
            score = 0
            if "return_mean" in base_lower:
                score += 3
            if "mean" in base_lower:
                score += 1
            pair_candidates.append((score, base_lower, true_tag, pred_tag))

    if pair_candidates:
        pair_candidates.sort(key=lambda x: (-x[0], x[1]))
        _, _, true_tag, pred_tag = pair_candidates[0]
        return true_tag, pred_tag

    grouped = {}
    for tag in return_tags:
        lower = tag.lower()
        base = _normalize_base_tag(tag)
        if base not in grouped:
            grouped[base] = {"true": [], "pred": []}

        if any(k in lower for k in ["pred", "prediction"]):
            grouped[base]["pred"].append(tag)
        if any(k in lower for k in ["true", "target", "gt", "groundtruth"]):
            grouped[base]["true"].append(tag)

    pair_candidates = []
    for base, sides in grouped.items():
        if sides["true"] and sides["pred"]:
            true_tag = sorted(sides["true"])[0]
            pred_tag = sorted(sides["pred"])[0]
            score = 0
            if "return_mean" in base:
                score += 2
            if "mean" in base:
                score += 1
            pair_candidates.append((score, base, true_tag, pred_tag))

    if not pair_candidates:
        return None, None

    pair_candidates.sort(key=lambda x: (-x[0], x[1]))
    _, _, true_tag, pred_tag = pair_candidates[0]
    return true_tag, pred_tag


def _load_scalar_series(ea, tag, max_steps):
    scalar_data = ea.Scalars(tag)
    if max_steps is None:
        steps = [s.step for s in scalar_data]
        values = [s.value for s in scalar_data]
    else:
        step_limit = max_steps + 5000
        steps = [s.step for s in scalar_data if s.step <= step_limit]
        values = [s.value for s in scalar_data if s.step <= step_limit]
    return steps, values


def _aggregate_runs(all_steps, all_values, smooth_weight):
    if not all_values:
        return None

    min_len = min(len(v) for v in all_values)
    if min_len == 0:
        return None

    aligned_values = [[] for _ in range(min_len)]
    for values in all_values:
        for idx in range(min_len):
            aligned_values[idx].append(values[idx])

    median_values = [np.median(v) for v in aligned_values]
    min_values = [np.min(v) for v in aligned_values]
    max_values = [np.max(v) for v in aligned_values]

    median_values = smooth(median_values, smooth_weight)
    min_values = smooth(min_values, smooth_weight)
    max_values = smooth(max_values, smooth_weight)

    return all_steps[0][:min_len], median_values, min_values, max_values


def merge_return_true_pred_plot(
    map_name,
    algo_name,
    variant_name,
    smooth_weight=0.8,
    max_steps=2e6,
    output_name=None,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    variant_path = os.path.join(script_dir, map_name, algo_name, variant_name)

    if not os.path.isdir(variant_path):
        print(f"Error: Variant path does not exist: {variant_path}")
        return

    event_files = _event_files(variant_path)
    if not event_files:
        print(f"Error: No event files found under {variant_path}")
        return

    sample_ea = event_accumulator.EventAccumulator(
        event_files[0], size_guidance={"scalars": 0}
    )
    sample_ea.Reload()
    scalar_tags = sample_ea.Tags().get("scalars", [])
    true_tag, pred_tag = _detect_true_pred_pair(scalar_tags)

    if true_tag is None or pred_tag is None:
        return_tags = [t for t in scalar_tags if "return" in t.lower()]
        print("Error: Could not match true/pred return tags.")
        print(f"Available return tags: {return_tags}")
        return

    print(f"Selected tags -> true: {true_tag}, pred: {pred_tag}")

    true_steps_runs, true_values_runs = [], []
    pred_steps_runs, pred_values_runs = [], []

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                event_file, size_guidance={"scalars": 0}
            )
            ea.Reload()
            available_tags = ea.Tags().get("scalars", [])
            if true_tag not in available_tags or pred_tag not in available_tags:
                continue

            t_steps, t_values = _load_scalar_series(ea, true_tag, max_steps)
            p_steps, p_values = _load_scalar_series(ea, pred_tag, max_steps)

            if t_steps and p_steps:
                true_steps_runs.append(t_steps)
                true_values_runs.append(t_values)
                pred_steps_runs.append(p_steps)
                pred_values_runs.append(p_values)
        except Exception as exc:
            print(f"Warning: failed to read {event_file}: {exc}")

    true_stats = _aggregate_runs(true_steps_runs, true_values_runs, smooth_weight)
    pred_stats = _aggregate_runs(pred_steps_runs, pred_values_runs, smooth_weight)

    if true_stats is None or pred_stats is None:
        print("Error: Not enough valid runs for true/pred tags.")
        return

    t_steps, t_median, t_min, t_max = true_stats
    p_steps, p_median, p_min, p_max = pred_stats

    plt.figure(figsize=(7, 6))
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5

    plt.plot(t_steps, t_median, color="#017D6F", linewidth=2.5, label="true")
    plt.fill_between(t_steps, t_min, t_max, color="#90EE90", alpha=0.35)

    plt.plot(p_steps, p_median, color="#CD5C5C", linewidth=2.5, label="pred")
    plt.fill_between(p_steps, p_min, p_max, color="#FA8072", alpha=0.35)

    plt.xlabel("训练时间步", fontsize=15)
    plt.ylabel("Return Mean", fontsize=15)
    plt.title(f"{map_name}-{algo_name}-{variant_name}: true vs pred", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True)

    if output_name is None:
        output_name = f"{map_name}_{algo_name}_{variant_name}_return_true_pred.png"
        output_name = output_name.replace(os.sep, "_")

    save_path = os.path.join(script_dir, output_name)
    plt.savefig(save_path)
    print(f"Merged figure saved to: {save_path}")


if __name__ == "__main__":
    # 示例：compare/地图名/算法名/变体名/.../events.out.tfevents*
    # 例如：compare/3s5z/QMIX/residual/<run>/events.out.tfevents*
    merge_return_true_pred_plot(
        map_name="MMM2",
        algo_name="res_multi_with_0.5reg",
        variant_name="1",
        smooth_weight=0.8,
        max_steps=2.6e6,
    )

    merge_return_true_pred_plot(
        map_name="MMM2",
        algo_name="res_multi_with_0.5reg",
        variant_name="2",
        smooth_weight=0.8,
        max_steps=2.6e6,
    )

    merge_return_true_pred_plot(
        map_name="MMM2",
        algo_name="res_multi_with_0.5reg",
        variant_name="3(best)",
        smooth_weight=0.8,
        max_steps=2.6e6,
    )
