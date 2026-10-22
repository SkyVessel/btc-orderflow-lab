# 数据获取与复现范围

当前权重可离线加载；22项离线测试不需要行情或账户。`tools/check_release.py`检查原v0.2文件/研究源代码哈希，并检查权重推理。

原始行情未发布：公开接口可读取不自动授予再分发权。`provenance`保存来源、采集时间和原文件哈希。旧报告中的原始数据相对路径表示原研究环境，克隆后并不存在对应原始文件。

## 获取下一轮数据

`python tools/download_public.py --output data/new-snapshot`从Yahoo取近期BITX五分钟数据，从Binance market-data-only域名取对应BTCUSDT五分钟数据（含真实主动买入量），保存本地原响应、URL、获取时间和哈希。网络代理出错且允许直连时，可加`--direct-network`，不修改系统代理。

下载器排除当前未结束交易日、缺失和非完整390分钟交易日，排除名单进入manifest；下一轮历史基线只评价这些完整交易日。前瞻的应交易日和故障日需要另外按交易所日历统计，不能照搬该排除方法。

`python tools/train_baseline.py --data data/new-snapshot --output results/new-baseline`使用新快照运行v0.2结构的scratch基线：按时间60%/20%/20%切分，3个固定阈值，先锁定验证选择再评价尾段，保留全部候选与成本账本。它不复用小时预训练、不改变原权重、没有自动优化或前瞻交易。超过API历史保留期的数据未必能补回。

任何新下载都是当前修订版本；即使日期相同，也要记录价格/分红调整和哈希差异。该基线的末段默认标为历史研究，不声称独立未见留出。

## 精确复现旧实验

原`v2/run_v2.py`依赖旧`data/bitx_5m.csv.gz`、`data/btc_5m.csv.gz`、原`data/manifest.json`及`data/v0.2/raw/`四份Yahoo响应。路径和哈希见provenance。必须取得并验证原快照，不能拿新下载冒充。`v2/verify_v2.py`还需要第二个完整复现目录；公开包不包含它。仓库保存的`v0.2-verification.json`是原环境的验证证据，新的云端不能仅因该文件存在而宣称重跑通过。

## 实时数据缺口

目前无BITX已验证实时bid/ask，没有实时执行器。Yahoo的延迟、断流、修订必须实测；不能用最新显示价假定可成交。BTC公开成交/盘口仅代表该交易所。Alpaca免费IEX数据覆盖单一交易所、需账户密钥；当前未接入。免费公开数据路线先继续，付费或账户接入需独立授权。
