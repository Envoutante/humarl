
# 快速上手 analyse.py

**`analyse.py`** 帮助我们把多次实验结果 (即 TensorBoard 事件文件)    汇总成一张对比图：

- 它会将每个 **算法变体** 的多次实验结果汇总成一条曲线；
- 并在一张图中绘制出这些算法变体所对应的曲线，以进行对比；
- 最后把绘制结果保存为 PNG (该方法适用于 SSH 服务器等无 GUI 的环境)   。

> 算法变体：是指**在经典算法/基线算法之上做了改动后的版本**。例如：采用了不同的奖励建模、损失函数、网络结构、超参数等。

<br>
<br>


## 第一部分：如何使用 (Step-by-step) 

### Step 1：把实验结果按固定目录结构放好

**`analyse.py` 所在的目录** 被视作根目录 (记为 `<analysis_root>`)   。

你需要把实验结果按如下结构组织：

```
<analysis_root>/
    <地图名1>/
        <算法名1>/
            <算法变体名A>/
                <run1>/
                    events.out.tfevents....
                <run2>/
                    events.out.tfevents....
            <算法变体名B>/
                <run1>/
                    events.out.tfevents....
        <算法名2>/
    <地图名2>/
        <算法名1>/
```

约定含义：
- `<地图名>`：例如 `MMM2`。
- `<算法名>`：例如 `QMIX`。
- `<算法变体名>`：你希望对比的 “改动版本” (每个变体对应一条曲线) 。
- `<runX>`：同一变体的多次独立重复实验 (不同 seed) ，用于做结果汇总。

脚本通过文件名前缀 `events.out.tfevents` 识别 TensorBoard 事件文件 (会递归扫描每个 run 目录) 。

**(可选)** 如果你在目录里也放了 `.npy`，脚本会尝试读取 (具体支持格式见 `load_npy_data`) 。

<br>

### Step 2：运行脚本

绘图调用位于 **`analyse.py`** 的尾部，只需要改动这一行：

```
plot_map_algorithms("MMM2/QMIX", "test_battle_won_mean", 0.8)
```

参数说明：
- `"MMM2/QMIX"`：`"<地图名>/<算法名>"`
- `"test_battle_won_mean"`：TensorBoard scalar 名
- `0.8`：平滑系数 (指数平滑；越接近 1 越平滑) 

在项目根目录运行：

```
python <analysis_root>/analyse.py
```

<br>

### Step 3：查看输出

**`analyse.py`** 会保存 PNG 到 `<analysis_root>/`：
- 文件名：`<地图名>_<算法名>_win_rate.png`
- 终端会打印保存路径：`Plot saved to ...`


<br>
<br>


## 第二部分：这个文件做了什么

**`analyse.py`** 分别对每个算法变体的实验结果进行处理，然后绘制到同一张图上。

**`analyse.py`** 的具体操作如下：

1. 递归收集该变体下的所有 `events.out.tfevents*` (多次 run) 。
2. 读取你指定的 `scalar_name`；如果该标量不存在，会回退尝试 `eval_win_rate`、`incre_win_rate`。
3. 对齐不同 run 的序列长度：以最短序列为准进行截断。
4. 逐时间步做统计汇总：
   - median 作为主曲线：对多次 run 的数值取中位数
   - min/max 作为阴影带 (体现 run 间波动范围) 
5. 对曲线与阴影带做指数平滑 (`smooth_weight`) 。
6. 额外在终端打印一个“末段统计” (方便论文表格的制作)：
   - 最后 250k 步窗口内，“中位数曲线”的最高点 (对时间取最大值) ：反映后期最好表现
   - 各 run 在最后 250k 步窗口内“峰值”的标准差：反映不同 run/seed 的波动大小 (std 越小通常越稳定) 

<br>
<br>

---

**END**