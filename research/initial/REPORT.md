# 历史回放结果

这是分钟级模拟结果，币种为 USDT；真实成交收益为空。

数据区间：2026-08-23 至 2026-10-21，86,400 根完整分钟 K 线。
训练/验证/留出：30/15/15 天。

| 验证集候选 | 成交数 | 净收益 USDT |
|---|---:|---:|
| breakout | 0 | 0.00 |
| flow_pullback | 0 | 0.00 |
| sweep_reclaim | 0 | 0.00 |

选定：breakout。留出净收益 0.00 USDT，成交 0 笔。
滑点加倍：0.00；纯价格基准：-361.16。
自动主账本准入：未通过，主账本保持现金；按用户新要求，仅采集数据，不运行失败候选的前瞻交易。

检查：
{
  "validation_net_positive": false,
  "validation_sample": false,
  "test_sample": false,
  "test_net_positive": false,
  "profit_factor": false,
  "daily_ci_positive": false,
  "stress_positive": false,
  "beats_price_baseline": true,
  "drawdown": true
}

限制：
- No historical depth/queue or individual-trade-size features
- Stop first if stop and target occur in the same minute
- Historical spread/slippage are assumptions; live fills use snapshots
- Daily bootstrap is exploratory, not proof of stationarity
- Drawdown measured at exits; intratrade drawdown may be larger
