# BITX 模型：训练、回测与纸面交易研究

当前标的为 **BITX（两倍日收益 BTC ETF）**。账户现金全额买入、只做多；初始模拟资金 **10,000 美元**，持有 **5–30 分钟**，当日平仓；目标是 **每日扣除交易成本后净赚 200 美元**。无融资、卖空、实盘接口、入金或每日本金重置。

**当前模型 bitx-v0.2 未达到盈利目标。** 最后12个历史交易日36笔，净亏153.08美元，达标0/12；其中17笔是预先登记的截止点探索交易。模型不是自动盈利系统。用户现已授权由 Dot 接手持续训练、优化及前瞻纸面实验，包括对失败版本的研究性模拟。

## 给 Dot 的入口

**从 [DOT_HANDOFF.md](DOT_HANDOFF.md) 开始。** 它包含用户的新授权、持续迭代任务、验收标准、账本要求和现有缺口。仓库已整理，不代表云端进程或自动任务已经启动。当前仓库提供历史研究与模型文件；实时纸面执行和持续调度由 Dot 在接手后实现、检验并启动。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/check_release.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s v2/tests -v
```

## 文件地图

- 当前选定权重：`results/v0.2/scratch_region.npz`、`scratch_returns.npz`；阈值与权重哈希：`selection_before_test.json`。
- 区域识别与特征：`v2/regions.py`；现金、结算、成本、交易计划：`v2/execution.py`。
- 原训练流程：`v2/run_v2.py`；训练协议：`v2/protocol.json`。
- [v0.2 完整报告](results/v0.2/REPORT.md)、[每日账本](results/v0.2/test_daily.csv)、[逐笔账本](results/v0.2/test_trades.csv)、[全部候选](results/v0.2/all_validation_trials.json)。
- [失败原因与工程缺口](docs/WHY_NOT_PROFITABLE.md)、[数据获取与历史复现](docs/DATA.md)。
- v0.1 代码与结果保留，用于核对旧版本；默认使用 v0.2。旧 BTC 项目已从默认分支替换，保留在标签 `archive/btc-before-bitx-2026-10-22`。

模型共2,852个参数，选定权重约26KB；普通CPU足以运行。原始行情、私人笔记和本机运行数据库未上传。公开来源及原哈希在 `provenance/`；新下载数据不自动等于原历史快照。

## 新数据上的基线运行

```sh
.venv/bin/python tools/download_public.py --output data/new-snapshot
.venv/bin/python tools/train_baseline.py --data data/new-snapshot --output results/new-baseline
```

这两个命令给 Dot 提供获取近期免费数据、运行既有结构基线的入口。它们不会自动改进策略，也不会启动前瞻交易。每次使用新的输出目录。旧 `v2/run_v2.py` 需要原始快照及其目录结构；不能在无历史数据的克隆中声称已精确复现旧成绩。

收益金额、目标达标率与交易活跃度分别报告。平均少于2笔/日或活跃日不足80%均失败；强制探索单独记账。有交易、测试通过、单日偶然盈利，都不等于完成盈利目标。
