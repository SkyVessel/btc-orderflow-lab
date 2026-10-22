# BTC Orderflow Lab

用免费公开数据研究比特币订单流，自动运行本地模拟交易，记录成本与失败结果。

**当前：btc-v0.2 已完成训练；仍未通过盈利准入。行情运行器只采集，不执行失败策略。**
这里没有 API 密钥、账户登录或真实下单接口。未通过历史准入的版本只采集数据、记录候选信号，不进行模拟交易。
Fabio Valentini 的拍卖市场与订单流方法提供研究假设；本项目是自行量化的版本，不代表其原版策略。

## 当前结果

第二版扩大到 180 天新历史数据、14 个特征、正则逻辑回归及独立概率校准，共比较 12 个固定组合。Brier 误差比固定概率基准低约 2.8%，但没有合格交易。详情见 [v0.2 训练报告](research/v0.2/REPORT.md)。

以下为首版记录：

- 真实数据：Binance BTC/USDT，2026-08-23 至 2026-10-21，86,400 根完整分钟 K 线。
- 时间划分：30 天训练、15 天验证选择、15 天一次性留出评价。
- 初始模拟资金：每个独立账本 10,000 USDT；现货做多、不借贷、不做空。
- 手续费：单边 0.10%，不假设 BNB 折扣或 VIP 资格。
- 首版三类模型均未过门槛，当前只采集行情及候选信号，不运行这些失败策略的前瞻交易。
- 结果见 [初始报告](research/initial/REPORT.md)。历史回放、实时模拟、真实支出分别记账。
- [事后诊断](research/initial/diagnostics-summary.json)记录未加概率过滤的候选与概率校准；[验证记录](research/initial/verification.json)包含 早期测试、历史逐笔对账与真实行情接入检查。

当前代码通过 29 项离线测试。测试验证程序与账本行为，不证明盈利。

## 快速开始（当前仅采集）

Python 3.11+，macOS 或 Linux。Windows 建议 WSL。无需注册交易账户。

```sh
git clone https://github.com/SkyVessel/btc-orderflow-lab.git
cd btc-orderflow-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m lab run --minutes 10
```

读取 `runtime/v0.1/STATUS.md` 和 `status.json`。它们显示交易准入、主账本、未平仓仓位及行情时间。当前 trading_enabled=false。
如当前环境的代理连接失败，并且允许直连官方公开接口，可以显式加 `--direct-network`。不修改系统代理。

持续运行及故障重启：

```sh
.venv/bin/python -m lab.supervisor --runtime runtime/v0.1
```

这条命令前台运行；Dot 应在能够持续存活的终端/服务中启动。一次聊天调用结束不保证进程继续存活。
Dot 完整交接见 [DOT_HANDOFF.md](DOT_HANDOFF.md)。本仓库不会自动创建外部定时任务。

## 暂停与恢复

```sh
touch runtime/v0.1/HALT
```

HALT 阻止新开仓，并在后续有效盘口、模拟延迟之后平仓。行情断开时仓位保持未解决，不能凭空按止损价成交。
确认状态全部平仓后，再终止 supervisor。删除 HALT 才恢复新开仓；Dot 不应未经实验主导者指示解除风险暂停。
直接停止进程会保留仓位，重启后按新盘口退出，缺口期间的价格风险计入模拟损益。

## 成本与收益

```sh
.venv/bin/python -m lab expense --id invoice-001 --usd 2.50 --category ai --note '本轮分摊'
.venv/bin/python -m lab report
.venv/bin/python -m lab export --output exports/day-01
```

- `realized_net_pnl`：已平仓模拟毛收益减双边手续费。滑点已经反映在成交价内，不重复扣除。
- `paid_simulated_fees`：包含已开仓尚未平仓的入场手续费。
- `recorded_real_expenses_usd`：实际登记的数据、AI、服务器等支出，以 USD 记录。
- `net_after_recorded_costs_usd`：模拟已实现净收益按显式 USDT/USD 假设换算，再减已登记支出。
- 未登记的 AI、电费、人工不是零成本。`cost_coverage=recorded_only` 明确表示成本尚不完整。
- `actual_exchange_trading_profit=null`：没有真实交易，不能将模拟结果称为真实到账利润。
- 现货 USDT 本位不等于美元，默认 1 USDT=1 USD 仅为报表换算假设，可用 `report --usd-per-usdt` 指定。

不要把实验资金重置或把负收益账本删掉。更改模型、费率、参数后使用新版本和新运行目录。

## 数学与数据

[模型协议](docs/MODEL.md) / [两周实验安排](docs/EXPERIMENT.md) / [数据来源](docs/DATA.md)

历史 API 的分钟总成交量 `V`、主动买入量 `B` 给出 `Delta=2B-V`。
模型采用按策略、Delta 强弱和趋势效率分组的收缩概率估计；样本不足或预期净收益不够时拒绝主账本交易。
实时逐笔大单、盘口失衡目前作为待研究变量记录，不假装历史已经具有这些字段。

## 重现历史实验

```sh
.venv/bin/python -m lab download --output data/history-new --days 60
.venv/bin/python -m lab research --data data/history-new --output research/new-version
```

下载按服务器时间取最近 60 个完整 UTC 日，因此以后运行的区间不同。初次数据请求 URL 和哈希保存在初始报告中；需要精确重现时按这些 URL 取回数据并校验。
不能用后来数据重跑覆盖初始结果，也不能把已经看过的日期重新称为未见测试。
历史运行会在本地生成 candidate_labels.jsonl；可用 `python -m lab.diagnose --research research/new-version --output research/new-version/diagnostics-summary.json` 重算事后诊断。它不修改模型或重新选择策略。
原始行情与运行数据库保留本地，不随公开仓库上传。GitHub 仅存代码、配置、模型与汇总证据。

## 运行边界

- 使用 Binance 官方 market-data-only 域名，无账户/交易能力；可访问公开数据不表示获得该地区的真实交易资格。
- 主动成交方向来自交易所字段；L2 是 20 档定期快照，不是全量 MBO，也不能精确归因每一次撤单。
- 盘口使用收到时刻，历史分钟回放使用固定价差假设。两种结果分开报告。
- 风险限制降低敞口，断线或跳价时不能保证损失恰好等于设定止损。
- 仓库公开不自动授予交易所数据的再分发许可；不要将原始行情批量推送到 GitHub。
