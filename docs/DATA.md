# 数据与费用来源

核实日期：2026-10-22。免费 API 的可访问性不代表真实交易资格。

| 用途 | 官方来源 |
|---|---|
| 公共只读 REST / WebSocket，无 API key | https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md |
| 分钟 K 线字段、主动买入量 | https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints |
| 逐笔成交和 20 档快照 | https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams |
| Binance 现货普通费率 0.100% / 0.100% | https://www.binance.com/en/fee/trading |
| Kraken 当前最低档 Spot Crypto 0.40% maker / 0.80% taker | https://www.kraken.com/features/fee-schedule |
| Fabio 公开方法 | https://worldclassedge.app/student-area/the-complete-order-flow/intro |

初期从 Kraken 切换为 Binance BTC/USDT，是因为当前公开现货费率更适合检验日内策略。
没有把 Binance 费率套在 Kraken 成交上，也没有把 USDT 当成无风险美元。
所有网络端点是公开行情域名；软件不接受交易密钥。

历史：87 个请求覆盖 60 个 UTC 日、86,400 根 1 分钟 K 线。保留每页原文、URL、采集时间和 SHA-256。
原始响应保存在本地 data/，发布汇总与哈希。数据修订时保留版本。
实时：订阅 trade、depth20@100ms、kline_1m，保存小时分割 gzip JSONL 原始消息与本地收到时间。
每分钟交易所完整 K 线用于策略，逐笔用于大单/CVD研究，20档快照用于模拟成交。
盘口快照没有交易所事件时间，收到时间是接入时点，不能据此精确测量交易所到本机延迟。
逐笔有事件时间和成交时间。跨流到达顺序可能不同，分钟日志中的逐笔汇总标注为“前次 K 线送达以来”。

Big Trade 是单笔成交额筛选，不是机构身份识别，也不是完整主动订单扫单重建。
不把 MBO、完整撤单、隐藏止损或全市场流动性标成已采集。

网络直连选项仅禁用本进程的环境代理，不切换地区、不绕过服务的访问限制。
若接口不向所在地区提供服务，停止并选择允许的数据源，重新登记标的/费率/数据协议。
