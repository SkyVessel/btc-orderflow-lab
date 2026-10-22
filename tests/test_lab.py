import copy
import json
import tempfile
import unittest
from pathlib import Path
from lab.core import Signals, cost_return, digest, fit, label, load_config, predict, replay, size
from lab.live import Book, PaperEngine, Store, report, single_runner

C = load_config(Path(__file__).parents[1] / "config.json")


def bar(t, o=100, h=101, l=99, c=100, v=10, buy=6):
    return dict(t=t, o=o, h=h, l=l, c=c, v=v, q=v*c, n=20, buy=buy)


def signal(t=60000):
    return dict(id="sample", strategy="breakout", signal_ms=t, reference=100,
                stop_fraction=.01, target_fraction=.02, cell="breakout|delta=1|trend=1",
                features=dict(delta_ratio=.4, efficiency=.5))


class ResearchTests(unittest.TestCase):
    def test_stress_ledger_prices_fees_and_pnl_reconcile(self):
        c = {**C, "hold_minutes": 2}
        future = [bar(60000, h=100.5, l=99.5), bar(120000, h=100.5, l=99.5)]
        normal, stressed = label(signal(), future, c), label(signal(), future, c, 2)
        row = {**signal(), **normal, "stress_label": stressed, "stress_net_return": stressed["net_return"]}
        result = replay([row], fit([], c, 0), "breakout", c, 0, 86400000, filtered=False, stress=True)
        trade = result["ledger"][0]
        self.assertEqual(trade["entry"], stressed["entry"])
        self.assertEqual(trade["exit"], stressed["exit"])
        self.assertAlmostEqual(trade["net_pnl"], trade["qty"] * (trade["exit"]-trade["entry"]) - trade["fees"])

    def test_both_barriers_stop_first_and_costs(self):
        c = {**C, "hold_minutes": 2, "slippage_bps_per_side": 0, "historical_half_spread_bps": 0}
        y = label(signal(), [bar(60000, h=104, l=98), bar(120000)], c)
        self.assertEqual(y["outcome"], "stop")
        self.assertAlmostEqual(y["net_return"], -.01 - .001 * 1.99)

    def test_gap_stop_uses_worse_open(self):
        c = {**C, "hold_minutes": 2, "slippage_bps_per_side": 0, "historical_half_spread_bps": 0}
        y = label(signal(), [bar(60000, h=100.5, l=99.5), bar(120000, o=95, h=96, l=94, c=95)], c)
        self.assertEqual(y["exit"], 95)

    def test_missing_or_unmatured_labels(self):
        self.assertIsNone(label(signal(), [bar(60000)], C))
        self.assertIsNone(label(signal(), [bar(120000)] * 60, C))

    def test_no_future_labels_in_training(self):
        r = {**signal(), "exit_ms": 120000, "net_return": .01, "outcome": "target"}
        model = fit([r], C, 119999)
        self.assertEqual(model["parents"]["breakout"]["n"], 0)
        self.assertFalse(predict(model, signal(), C)["eligible"])

    def test_probability_shrinkage_and_sum(self):
        rows = [{**signal(), "exit_ms": 120000, "net_return": .01 if i % 2 else -.003,
                 "outcome": "target" if i % 2 else "stop"} for i in range(60)]
        model = fit(rows, C, 120000)
        p = predict(model, signal(), C)
        self.assertAlmostEqual(p["p_profit"], .5)
        self.assertAlmostEqual(sum(p["p_outcomes"].values()), 1)
        self.assertAlmostEqual(p["expected_net_return"], .0035)

    def test_signal_prefix_is_causal(self):
        bars = [bar(i*60000, o=100+i*.1, h=101+i*.1, l=99+i*.1, c=100+i*.1) for i in range(100)]
        a, b = Signals(C), Signals(C)
        first = [a.push(x) for x in bars[:80]]
        full = [b.push(x) for x in bars]
        self.assertEqual(first, full[:80])
        for rows in first:
            for s in rows:
                self.assertGreater(s["signal_ms"], 0)

    def test_size_counts_fee_risk_and_cash(self):
        qty = size(10000, 10000, 100, .002, C)
        self.assertLessEqual(qty * 100 * (1+C["fee_rate_per_side"]), 10000)
        self.assertLessEqual(qty * 100, 2500 + 1e-9)
        self.assertLessEqual(qty * 100 * (.002+.002+.0004), 25.00001)

    def test_historical_volume_delta(self):
        from lab.core import kline
        b = kline([0, "100", "101", "99", "100", "10", 59999, "1000", 20, "7", "700", "0"])
        self.assertEqual((2*b["buy"]-b["v"])/b["v"], .4)


class LiveTests(unittest.TestCase):
    def test_failed_model_logs_candidates_without_paper_trading(self):
        self.engine.signals.push = lambda _: [signal()]
        self.engine.on_bar(bar(0), 61000)
        self.assertEqual(list(self.engine.wallets), ["primary"])
        self.assertIsNone(self.engine.wallets["primary"]["pending"])
        self.assertIsNone(self.engine.wallets["primary"]["position"])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 1)
        self.assertFalse(self.store.get("trading_enabled"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/"ledger.sqlite")
        self.c = {**C, "min_notional": 1, "qty_step": .00001}
        self.model = fit([], self.c, 0)
        self.engine = PaperEngine(self.c, self.model, self.store)

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def book(self, bid=100, ask=100.01, id=1, qty=1000):
        return dict(lastUpdateId=id, bids=[[str(bid), str(qty)]], asks=[[str(ask), str(qty)]])

    def queue(self, now=61000):
        w = self.engine.wallets["primary"]
        w["pending"] = dict(side="buy", ready_ms=now+1000, expires_ms=now+5000, signal=signal(), prediction={})
        self.engine.last_market_ms = now
        return w

    def test_latency_and_round_trip_reconciles(self):
        w = self.queue()
        self.engine.on_book(self.book(), 61500)
        self.assertIsNone(w["position"])
        self.engine.on_book(self.book(id=2), 62000)
        self.assertIsNotNone(w["position"])
        self.engine.on_book(self.book(bid=103, ask=103.01, id=3), 63000)
        self.assertEqual(w["pending"]["side"], "sell")
        self.engine.on_book(self.book(bid=103, ask=103.01, id=4), 64001)
        self.assertIsNone(w["position"])
        self.assertAlmostEqual(w["cash"] - 10000, w["realized_pnl"])
        trade = json.loads(self.store.db.execute("SELECT payload FROM trades").fetchone()[0])
        self.assertAlmostEqual(trade["gross_pnl"] - trade["entry_fee"] - trade["exit_fee"], trade["net_pnl"])

    def test_insufficient_depth_rejects(self):
        w = self.queue()
        self.engine.on_book(self.book(qty=.00001), 62000)
        self.assertIsNone(w["position"])
        self.assertEqual(w["cash"], 10000)

    def test_halt_cancels_pending_entry(self):
        w = self.queue()
        self.engine.halt = True
        self.engine.on_book(self.book(), 62000)
        self.assertIsNone(w["position"])

    def test_duplicate_book_and_trade(self):
        book = Book()
        self.assertTrue(book.update(self.book(), 1000))
        self.assertFalse(book.update(self.book(), 2000))
        msg = dict(t=100, E=1000, q="1", p="100", m=False)
        self.engine.trade(msg, 1000)
        self.engine.trade(msg, 1000)
        self.assertEqual(self.engine.session_cvd, 1)

    def test_stale_market_does_not_fill_entry(self):
        w = self.queue()
        self.engine.last_market_ms = 0
        self.engine.on_book(self.book(), 62000)
        self.assertIsNone(w["position"])

    def test_crossed_book_rejected(self):
        with self.assertRaises(ValueError):
            Book().update(self.book(bid=101, ask=100), 1000)

    def test_changed_source_is_part_of_fingerprint(self):
        self.assertIn("core.py", self.store.get("source_hashes"))

    def test_unknown_future_bar_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.on_bar(bar(60000), 60001)

    def test_restart_recovers_open_position(self):
        self.queue()
        self.engine.on_book(self.book(), 62000)
        restored = PaperEngine(self.c, self.model, self.store)
        w = restored.wallets["primary"]
        self.assertIsNotNone(w["position"])
        self.assertEqual(w["pending"]["reason"], "recovery_exit")
        restored.on_book(self.book(bid=90, ask=90.01), 90000)
        self.assertLess(w["realized_pnl"], 0)
        self.assertIsNone(w["position"])

    def test_config_change_cannot_overwrite_wallet(self):
        with self.assertRaises(ValueError):
            PaperEngine({**self.c, "fee_rate_per_side": .008}, self.model, self.store)

    def test_real_profit_null_and_expenses_separate(self):
        self.store.db.execute("INSERT INTO expenses VALUES('invoice',0,2.5,'ai','private note')")
        self.store.commit()
        r = report(self.store, self.temp.name)
        self.assertIsNone(r["actual_exchange_trading_profit"])
        self.assertEqual(r["wallets"]["primary"]["net_after_recorded_costs_usd"], -2.5)

    def test_second_runner_is_blocked(self):
        with single_runner(self.temp.name):
            with self.assertRaises(RuntimeError):
                with single_runner(self.temp.name):
                    pass


if __name__ == "__main__":
    unittest.main()
