# 给 Dot 的执行说明

v0.1、v0.2 和 v0.3 当前均未通过盈利准入。本版只能用于行情采集和候选信号记录，不安排失败策略的前瞻交易。研究主导者交付通过独立历史测试的新版本之后，再启动纸面交易验证。你负责运行与监管，不自行升级模型。

## 必须保持

- 仅模拟，不连接真实交易账户。无需密码、API key、入金或授权交易。
- 行情采集固定使用 config.json 与 research/initial/model.json（v0.1 格式）；v0.2 为研究模型，不兼容此执行器。主账本当前未过门槛，保持现金；不运行候选观察交易。
- 不改手续费、门槛或仓位，不删除亏损，不混合多个版本账本。
- 不将纸面成交称为交易所成交。真实运行支出与模拟手续费分开。

## 启动

先核实你能执行终端命令，并能维持一个长期进程。若只能浏览网页，先报告该具体限制；不要假装已经运行。

```sh
git clone https://github.com/SkyVessel/btc-orderflow-lab.git
cd btc-orderflow-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m lab run --runtime runtime/check --minutes 2
```

检查 `runtime/check/status.json`：应有新鲜 heartbeat、合理 bid<ask、无持续连接错误。
短检查的收益常为零；这是接入检查，不能作为策略评价。
如启动当前版本的持续数据采集，另开目录并在前台持久会话中运行：

```sh
.venv/bin/python -m lab.supervisor --runtime runtime/v0.1
```

如果环境必须在后台运行，可用其服务管理器保持该命令。日志/原始行情写入本地 runtime/v0.1。
不能只依赖短暂聊天工具会话。确保设备不休眠且磁盘有足够空间。
代理故障时可以在允许直连的环境显式加 --direct-network。

## 监管与停止

建议每 15 分钟检查一次。只有确认平台支持定时任务后才配置；本仓库本身不包含 Dot 的调度能力。

```sh
.venv/bin/python -m lab report --runtime runtime/v0.1
```

- heartbeat 超过 30 秒、持续断线超过 5 分钟、未解决退出订单超过 30 秒、异常负现金、账本不一致：报告并创建 `runtime/v0.1/HALT`。
- 收到 HALT 后，等待所有仓位退出；网络故障下仓位不能假设已经关闭。
- supervisor 自动重启同一版本，SQLite 与单进程锁防止重复运行。重启时未平仓按新盘口退出。
- 不通过换目录/重置钱包绕过当日风险限制。
- 升级前暂停并确认平仓，由主导者提供新模型/配置，再用新版本目录运行。

## 每日交付

```sh
.venv/bin/python -m lab export --runtime runtime/v0.1 --output exports/YYYY-MM-DD
```

交付 STATUS.md、status.json、decisions.jsonl、trades.jsonl、events.jsonl、fingerprint.txt。
采集阶段汇报数据缺口、运行时长、候选信号数和版本。通过准入并开始纸面交易后，再汇报已平仓毛/净收益、手续费、持仓浮盈亏和支出。
初始版本同样需要报告拒绝交易原因；不要只发送盈利交易。
需要给研究主导者原始数据时单独交付 runtime 原始行情，不自动公开再分发。

实际费用有依据后登记，示例：

```sh
.venv/bin/python -m lab expense --runtime runtime/v0.1 --id unique-invoice --usd 1.50 --category hosting --note '本轮分摊'
```

免费行情订阅为零，但 AI、电费和人工分摊未知时明确写未知。不要编造支出。
每 48–72 小时由用户把日志交回主导者；你不主动给其他会话发送消息，除非用户明确授权。

## 何时结束

用户要求停止时，或者到 2026-11-05（纽约时间）的评估截止日，先暂停新开仓，待持仓处理完后导出完整结果。
如果运行开始得晚，报告实际覆盖时长。到期没有盈利也要完整交付失败证据。
