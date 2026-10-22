import copy
import json
import math
import unittest
from pathlib import Path
import numpy as np
from lab.train_v2 import aggregate, fit_model, logistic, partition, replay, score, signals
from lab.core import label

P = json.loads((Path(__file__).parents[1]/"research/v0.2/protocol.json").read_text())


def bars(count):
    result = []
    for i in range(count):
        o = 100+i*.01
        result.append(dict(t=i*60000,o=o,h=o+.05,l=o-.03,c=o+.01,v=10 if i%5 else 30,
                           q=(10 if i%5 else 30)*o,buy=8 if i%5 else 25,n=2))
    return result


class V2Tests(unittest.TestCase):
    def test_aggregation_preserves_volume_and_last_close(self):
        source = bars(10)
        result = aggregate(source)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]["v"],sum(b["v"] for b in source[:5]))
        self.assertEqual(result[0]["c"],source[4]["c"])
        with self.assertRaises(ValueError):
            aggregate(source[1:])

    def test_signal_features_do_not_use_future_bars(self):
        source = bars(1000)
        # Create rising structure with a high-volume impulse in several closed bars.
        for i in range(300,1000,25):
            for b in source[i:i+5]:
                b["v"] *= 4
                b["q"] *= 4
                b["buy"] *= 4
        prefix = signals(source[:700],P)
        full = signals(source,P)
        self.assertTrue(prefix)
        self.assertEqual(prefix,[r for r in full if r["signal_ms"]<=700*60000])
        self.assertTrue(all(len(r["x"])==14 for r in full))

    def test_label_must_mature_within_partition(self):
        rows=[dict(signal_ms=100,exit_ms=300),dict(signal_ms=200,exit_ms=400)]
        self.assertEqual(partition(rows,0,350),rows[:1])

    def test_regularized_logistic_recovers_informative_direction(self):
        x=np.array([[1,i/10] for i in range(-20,21)])
        y=np.array([i>0 for i in range(-20,21)],dtype=float)
        beta=logistic(x,y,10)
        self.assertGreater(beta[1],0)
        self.assertTrue(np.isfinite(beta).all())

    def test_training_and_scoring_are_frozen(self):
        rows=[dict(strategy="impulse_breakout",x=[i%3,(i%7)/7]+[0]*12,
                   net_return=.01 if i%2 else -.006,stop_fraction=.005) for i in range(200)]
        model=fit_model(rows[:150],rows[150:],10)
        before=copy.deepcopy(model)
        a=score(rows[:5],model)
        b=score(rows,model)
        self.assertEqual(a,b[:5])
        self.assertEqual(before,model)
        self.assertTrue(all(0<r["p_profit"]<1 and math.isfinite(r["expected_net_return"]) for r in b))

    def test_replay_costs_and_stress_reconcile(self):
        s=dict(id="one",strategy="impulse_breakout",signal_ms=60000,stop_fraction=.01,target_fraction=.02,
               p_profit=.7,expected_net_return=.005)
        future=[dict(t=60000,o=100,h=100.2,l=99.8,c=100),dict(t=120000,o=100,h=100.2,l=99.8,c=100)]
        c=dict(hold_minutes=2,slippage_bps_per_side=2,historical_half_spread_bps=1,fee_rate_per_side=.001)
        row={**s,**label(s,future,c),"stress_label":label(s,future,c,2)}
        for stress in (False,True):
            result=replay([row],P,0,86400000,.0005,stress)
            t=result["ledger"][0]
            self.assertAlmostEqual(t["net_pnl"],t["qty"]*(t["exit"]-t["entry"])-t["fees"])
            self.assertAlmostEqual(result["gross_pnl"]-result["fees"],result["net_pnl"])
            self.assertEqual(result["daily_pnl"]["1970-01-01"],t["net_pnl"])

    def test_no_trade_is_zero_not_a_profitable_sample(self):
        result=replay([],P,0,3*86400000,.0005)
        self.assertEqual(result["trades"],0)
        self.assertEqual(result["mean_daily_pnl_ci95"],[0,0])


if __name__ == "__main__":
    unittest.main()
