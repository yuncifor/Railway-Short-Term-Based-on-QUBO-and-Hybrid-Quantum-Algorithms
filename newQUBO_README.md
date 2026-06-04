# newQUBO.PY README

## 中文说明

### 项目简介

`newQUBO.PY` 是一个基于 QUBO（Quadratic Unconstrained Binary Optimization，二次无约束二进制优化）的上下行列车调度优化实验程序。程序以 A 至 G 共 7 个车站、8 列上下行列车为基础场景，将列车发车顺序、区间轨道分配、换轨成本、区间运行时间波动和中间站停靠线容量限制等因素纳入统一能量函数，并通过多种经典、类量子和混合启发式算法搜索低能量调度方案。

该程序适合用于铁路调度优化、QUBO 建模、启发式算法对比、消融实验、鲁棒性实验和论文实验数据生成。

### 主要功能

- 构建列车调度问题的 QUBO 矩阵模型。
- 支持上下行列车在多区间、多站点上的发车顺序和轨道分配优化。
- 计算调度方案的总能量、总延误、换轨成本、区间波动惩罚和中间站停靠线惩罚。
- 支持 10 种 QUBO 求解方法，包括暴力遍历、遗传算法、粒子群、模拟退火、量子启发式算法和 QAOA 混合算法。
- 提供单次求解结果展示和算法迭代曲线查看。
- 支持区间波动实验、多算法统一对比、消融实验、QUBO 权重搜索、鲁棒性实验、收敛性实验、权重敏感性实验、问题规模实验和场景参数实验。
- 可导出实验数据、统计结果、Markdown 报告和图表。
- 可导出当前最优调度方案的列车时刻表和运行图。

### 文件结构

```text
newQUBO.PY                 主程序文件
newQUBO_README.md          本说明文件
experiment_exports/        程序运行后自动生成的实验导出目录
```

说明：`experiment_exports` 是相对运行目录创建的。如果在程序所在目录运行，导出目录会自动生成在该目录下。

### 运行环境

建议使用 Python 3.8 或更高版本。

程序依赖以下 Python 库：

- `numpy`
- `matplotlib`

程序还使用 Python 标准库：

- `random`
- `time`
- `math`
- `csv`
- `json`
- `os`
- `itertools`

安装依赖：

```bash
pip install numpy matplotlib
```

### 运行方法

在命令行进入 `newQUBO.PY` 所在目录后运行：

```bash
cd path/to/newQUBO-folder
python newQUBO.PY
```

如果系统中 `python` 命令不可用，可尝试：

```bash
py newQUBO.PY
```

### 主菜单说明

程序启动后进入交互式主菜单：

| 选项 | 功能 |
| --- | --- |
| 1 | 选择 QUBO 算法并执行单次求解 |
| 2 | 显示当前最优调度结果 |
| 3 | 查看算法迭代能量变化图 |
| 4 | 运行区间波动实验 |
| 5 | 设置系统参数 |
| 6 | 进入高级实验菜单 |
| 7 | 导出列车时刻表与运行图 |
| 0 | 退出程序 |

### 支持的求解算法

| 编号 | 算法名称 |
| --- | --- |
| 1 | QUBO-暴力遍历算法 |
| 2 | QUBO-传统遗传算法 |
| 3 | QUBO-传统粒子群算法 |
| 4 | QUBO-传统模拟退火算法 |
| 5 | QUBO-量子遗传算法 |
| 6 | QUBO-量子粒子群算法 |
| 7 | QUBO-量子模拟退火算法 |
| 8 | QUBO-量子遗传-QAOA混合算法 |
| 9 | QUBO-量子退火-QAOA混合算法 |
| 10 | QUBO-量子粒子群-QAOA混合算法 |

注意：程序中的“量子”和“QAOA”算法为量子启发式或局部搜索模拟实现，不依赖真实量子硬件或专用量子计算框架。

### 调度目标与能量函数

程序将调度方案质量综合表示为能量值，能量值越低表示方案越优。主要组成包括：

- 总延误时间：包含始发站延误和中间站延误。
- 换轨总成本：列车在相邻区间轨道变化时产生的惩罚。
- 区间波动总惩罚：区间运行时间随机波动时产生的惩罚。
- 中间站停靠线总惩罚：中间站同时停靠列车超过正常停靠线数量时产生的惩罚。

总能量的核心形式为：

```text
总能量 = 总延误 + 换轨总成本 + 区间波动总惩罚 + 中间站停靠线总惩罚
```

### 关键参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `FLUCTUATION_ENABLE` | `False` | 是否启用区间运行时间随机波动 |
| `FLUCTUATION_RATE` | `0.2` | 区间运行时间波动幅度，默认 ±20% |
| `FLUCTUATION_PENALTY_WEIGHT` | `1.5` | 区间波动惩罚权重 |
| `PLATFORM_CAPACITY_ENABLE` | `True` | 是否启用中间站停靠线容量限制 |
| `MAX_NORMAL_PLATFORMS` | `3` | 中间站正常可用停靠线数量 |
| `PLATFORM_PENALTY_WEIGHT` | `3.0` | 超出正常停靠线数量时的惩罚权重 |
| `track_change_cost` | `2` | 换轨惩罚成本 |
| `QUBO_ENERGY_SCALE` | `100.0` | QUBO 能量缩放系数 |

QUBO 权重默认配置：

| 权重项 | 默认值 | 含义 |
| --- | --- | --- |
| `ORDER_MUTEX_PENALTY` | `100.0` | 发车顺序互斥约束权重 |
| `TRACK_MUTEX_PENALTY` | `100.0` | 轨道选择互斥约束权重 |
| `CONFLICT_PENALTY` | `200.0` | 同区间同轨道冲突约束权重 |
| `DELAY_PENALTY` | `1.0` | 延误目标权重 |
| `TRACK_CHANGE_PENALTY` | `2.0` | 换轨目标权重 |

### 高级实验菜单

高级实验菜单提供以下功能：

| 选项 | 功能 |
| --- | --- |
| 1 | 统一算法对比实验 |
| 2 | 消融实验 |
| 3 | QUBO 权重搜索 |
| 4 | 扰动鲁棒性复现实验 |
| 5 | 收敛性与稳定性实验 |
| 6 | QUBO 权重局部敏感性实验 |
| 7 | 不同问题规模实验 |
| 8 | 场景参数实验 |
| 9 | 查看最近权重搜索结果 |
| 10 | 所有算法统一消融实验 |
| 11 | 所有算法统一 QUBO 权重搜索 |

### 输出结果

程序会根据所选功能输出不同类型的结果：

- 控制台输出：单次求解过程、最优能量、发车顺序、各列车到发时刻、延误与惩罚指标。
- CSV 文件：实验汇总表、详细实验记录、当前调度时刻表。
- JSON 文件：实验元数据、汇总结果和详细结果。
- Markdown 文件：实验报告和可直接写入论文的结论文字稿。
- PNG 图像：算法对比图、消融实验图、鲁棒性图、收敛性图、权重敏感性图和列车运行图。

常见导出文件命名示例：

```text
experiment_exports/统一算法对比实验_YYYYMMDD_HHMMSS_summary.csv
experiment_exports/统一算法对比实验_YYYYMMDD_HHMMSS.json
experiment_exports/统一算法对比实验报告_YYYYMMDD_HHMMSS.md
experiment_exports/当前列车运行图_YYYYMMDD_HHMMSS.png
```

### 使用建议

- 初次运行建议先选择 `1` 执行单次求解，再选择 `2` 查看结果。
- 如果需要图形化查看迭代过程，先运行算法，再选择 `3` 查看迭代图。
- 如果需要运行区间波动实验，应先在 `5. 设置系统参数` 中开启区间波动，或在源代码中将 `FLUCTUATION_ENABLE` 改为 `True`。
- 暴力遍历算法设置了最大采样次数，耗时可能较长；高级实验中通常建议先不包含暴力遍历算法。
- 高级实验会多次调用求解器，运行时间可能较长，建议先使用较小的重复次数测试。
- 如果 Matplotlib 中文显示异常，请确认系统中安装了 `SimHei`、`WenQuanYi Micro Hei` 或 `Heiti TC` 等中文字体。

### 注意事项

- 程序为交互式命令行程序，需要根据菜单输入数字选项。
- 部分实验包含随机过程，程序会通过随机种子提高复现实验的一致性，但不同环境下仍可能存在细微差异。
- `FLUCTUATION_ENABLE` 默认为 `False`，因此区间波动实验菜单在默认状态下会提示无法运行。
- 程序中的 QUBO 模型与调度能量函数存在缩放关系，`QUBO_ENERGY_SCALE` 用于将 QUBO 能量缩放到调度指标量级。
- 导出目录使用相对路径，建议在 `newQUBO.PY` 所在目录运行程序，便于管理输出文件。

---

## English Description

### Overview

`newQUBO.PY` is an interactive railway scheduling optimization and experiment script based on QUBO, short for Quadratic Unconstrained Binary Optimization. The default case contains 7 stations from A to G and 8 trains running in both directions. The program models departure order, interval track assignment, track switching cost, interval running time fluctuation, and intermediate-station platform capacity as a unified energy minimization problem.

The script is suitable for railway scheduling optimization studies, QUBO modeling, heuristic algorithm comparison, ablation studies, robustness experiments, convergence analysis, and paper-oriented experimental data generation.

### Main Features

- Builds a QUBO matrix for the train scheduling problem.
- Optimizes train departure order and track assignments across multiple stations and intervals.
- Computes total energy, total delay, track switching cost, fluctuation penalty, and platform capacity penalty.
- Supports 10 QUBO solvers, including brute-force sampling, genetic algorithm, particle swarm optimization, simulated annealing, quantum-inspired algorithms, and QAOA hybrid methods.
- Provides single-run scheduling display and iteration energy plots.
- Supports fluctuation experiments, unified algorithm comparison, ablation experiments, QUBO weight search, robustness experiments, convergence experiments, weight sensitivity analysis, problem-scale experiments, and scenario parameter experiments.
- Exports experiment data, statistics, Markdown reports, and figures.
- Exports the current best timetable and train running diagram.

### Files

```text
newQUBO.PY                 Main program
newQUBO_README.md          This README file
experiment_exports/        Export directory generated after running experiments
```

Note: `experiment_exports` is created relative to the current working directory. If the script is launched from the directory that contains `newQUBO.PY`, the export directory will be generated there automatically.

### Requirements

Python 3.8 or later is recommended.

Required third-party packages:

- `numpy`
- `matplotlib`

Standard-library modules used by the script:

- `random`
- `time`
- `math`
- `csv`
- `json`
- `os`
- `itertools`

Install dependencies:

```bash
pip install numpy matplotlib
```

### How to Run

Open a terminal and change to the directory that contains `newQUBO.PY`:

```bash
cd path/to/newQUBO-folder
python newQUBO.PY
```

If `python` is not available on your system, try:

```bash
py newQUBO.PY
```

### Main Menu

After startup, the script opens an interactive menu:

| Option | Function |
| --- | --- |
| 1 | Select a QUBO solver and run a single optimization |
| 2 | Display the current best scheduling result |
| 3 | View algorithm iteration plots |
| 4 | Run interval fluctuation experiments |
| 5 | Set system parameters |
| 6 | Open the advanced experiment menu |
| 7 | Export timetable and train running diagram |
| 0 | Exit |

### Supported Solvers

| No. | Solver |
| --- | --- |
| 1 | QUBO brute-force traversal algorithm |
| 2 | QUBO classical genetic algorithm |
| 3 | QUBO classical particle swarm algorithm |
| 4 | QUBO classical simulated annealing algorithm |
| 5 | QUBO quantum genetic algorithm |
| 6 | QUBO quantum particle swarm algorithm |
| 7 | QUBO quantum simulated annealing algorithm |
| 8 | QUBO quantum genetic and QAOA hybrid algorithm |
| 9 | QUBO quantum annealing and QAOA hybrid algorithm |
| 10 | QUBO quantum particle swarm and QAOA hybrid algorithm |

Note: The “quantum” and “QAOA” methods in this script are quantum-inspired or local-search simulations. They do not require real quantum hardware or specialized quantum-computing libraries.

### Objective And Energy

The script evaluates scheduling quality by an energy value. A lower energy value means a better scheduling solution. The energy consists of:

- Total delay, including origin-station delay and intermediate-station delay.
- Track switching cost, caused by track changes between adjacent intervals.
- Interval fluctuation penalty, caused by random running-time fluctuation.
- Intermediate-station platform penalty, caused when simultaneous train occupation exceeds normal platform capacity.

Core energy form:

```text
Total energy = Total delay + Track switching cost + Fluctuation penalty + Platform capacity penalty
```

### Key Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `FLUCTUATION_ENABLE` | `False` | Enables random interval running-time fluctuation |
| `FLUCTUATION_RATE` | `0.2` | Running-time fluctuation range, default ±20% |
| `FLUCTUATION_PENALTY_WEIGHT` | `1.5` | Weight of fluctuation penalty |
| `PLATFORM_CAPACITY_ENABLE` | `True` | Enables intermediate-station platform capacity constraint |
| `MAX_NORMAL_PLATFORMS` | `3` | Number of normally available platforms at intermediate stations |
| `PLATFORM_PENALTY_WEIGHT` | `3.0` | Penalty weight for exceeding normal platform capacity |
| `track_change_cost` | `2` | Track switching penalty cost |
| `QUBO_ENERGY_SCALE` | `100.0` | QUBO energy scaling factor |

Default QUBO weights:

| Weight | Default | Meaning |
| --- | --- | --- |
| `ORDER_MUTEX_PENALTY` | `100.0` | Departure-order mutual-exclusion penalty |
| `TRACK_MUTEX_PENALTY` | `100.0` | Track-selection mutual-exclusion penalty |
| `CONFLICT_PENALTY` | `200.0` | Same-interval same-track conflict penalty |
| `DELAY_PENALTY` | `1.0` | Delay objective weight |
| `TRACK_CHANGE_PENALTY` | `2.0` | Track switching objective weight |

### Advanced Experiments

The advanced experiment menu supports:

| Option | Function |
| --- | --- |
| 1 | Unified algorithm comparison experiment |
| 2 | Ablation experiment |
| 3 | QUBO weight search |
| 4 | Disturbance robustness reproduction experiment |
| 5 | Convergence and stability experiment |
| 6 | QUBO weight local sensitivity experiment |
| 7 | Problem-scale experiment |
| 8 | Scenario parameter experiment |
| 9 | View the latest weight search result |
| 10 | Unified ablation experiment for all algorithms |
| 11 | Unified QUBO weight search for all algorithms |

### Outputs

Depending on the selected function, the script can generate:

- Console output: optimization progress, best energy, departure order, arrival and departure times, delays, and penalties.
- CSV files: summary tables, detailed experiment records, and the current timetable.
- JSON files: metadata, summary results, and detailed results.
- Markdown files: experiment reports and paper-ready conclusion drafts.
- PNG images: comparison charts, ablation charts, robustness charts, convergence charts, sensitivity charts, and train running diagrams.

Example export names:

```text
experiment_exports/统一算法对比实验_YYYYMMDD_HHMMSS_summary.csv
experiment_exports/统一算法对比实验_YYYYMMDD_HHMMSS.json
experiment_exports/统一算法对比实验报告_YYYYMMDD_HHMMSS.md
experiment_exports/当前列车运行图_YYYYMMDD_HHMMSS.png
```

### Usage Tips

- For a first run, choose `1` to execute a single optimization, then choose `2` to inspect the result.
- To view iteration plots, run a solver first and then choose `3`.
- To run interval fluctuation experiments, enable fluctuation from `5. Set system parameters`, or set `FLUCTUATION_ENABLE` to `True` in the source code.
- Brute-force traversal may take longer because it performs many random valid-solution evaluations. For advanced experiments, it is usually better to exclude brute force at first.
- Advanced experiments call solvers repeatedly and may take a long time. Start with a small number of trials.
- If Chinese text in Matplotlib figures is not displayed correctly, make sure Chinese fonts such as `SimHei`, `WenQuanYi Micro Hei`, or `Heiti TC` are installed.

### Notes

- The program is an interactive command-line script. Enter numeric options according to the menu prompts.
- Some experiments involve randomness. The script uses random seeds to improve reproducibility, but small differences may still appear across environments.
- `FLUCTUATION_ENABLE` is `False` by default, so the fluctuation experiment menu will report that fluctuation experiments cannot run until it is enabled.
- The QUBO model energy is scaled by `QUBO_ENERGY_SCALE` to align with the scheduling metric scale.
- Because export paths are relative, it is recommended to run the script from the directory that contains `newQUBO.PY`.
