import copy
import json
import unittest
from pathlib import Path
from lab.absorption import signals
from lab.audit_flow import summarize_trades
from lab.train_v3 import partition,select_trial,checks_for

P=json.loads((Path(__file__).parents[1]/"research/v0.3/protocol.json").read_text())


def source():
    rows=[]
    for i in range(330):
        b=dict(t=i*60000,o=100,h=100.1,l=99.9,c=100,v=10,buy=5,q=1000,n=10)
        if 300<=i<305:
            b.update(o=100,h=100.1,l=99.8,c=100.05,v=40,buy=8,q=4000)
        if 305<=i<310:
            b.update(o=100.05,h=100.3,l=100,c=100.2,v=15,buy=12,q=1502)
        rows.append(b)
    return rows


class AbsorptionTests(unittest.TestCase):
    def test_waits_for_later_bar_and_features_are_causal(self):
        bars=source()
        self.assertEqual(signals(bars[:305],P),[])
        early=signals(bars[:310],P)
        self.assertEqual(len(early),2)
        self.assertTrue(all(r["signal_ms"]==310*60000 for r in early))
        self.assertTrue(all(r["event_end_ms"]<r["signal_ms"] and len(r["x"])==19 for r in early))
        self.assertEqual(early,[r for r in signals(bars,P) if r["signal_ms"]<=310*60000])

    def test_new_low_before_confirmation_invalidates_event(self):
        rows=source()[:310]
        for b in rows[305:310]:b["l"]=99
        self.assertEqual(signals(rows,P),[])

    def test_active_buying_is_not_sell_absorption(self):
        rows=source()[:310]
        for b in rows[300:305]:b["buy"]=35
        self.assertEqual(signals(rows,P),[])

    def test_purge_full_horizon_even_if_stop_happened_early(self):
        row=dict(signal_ms=60000,exit_ms=120000,stress_label=dict(exit_ms=180000))
        self.assertEqual(partition([row],0,300000,120),[])
        self.assertEqual(partition([row],0,121*60000,120),[row])

    def test_cash_does_not_win_over_traded_diagnostic(self):
        trials=[dict(validation=dict(trades=0,net_pnl=0)),dict(validation=dict(trades=25,net_pnl=-100))]
        self.assertIs(select_trial(trials,P),trials[1])

    def test_aggregate_side_and_exact_volume(self):
        rows=[dict(a=1,T=1000,q="2.5",m=True,f=10,l=11),dict(a=2,T=2000,q="1.0",m=False,f=12,l=12)]
        s=summarize_trades(rows,0,60000)
        self.assertEqual(s["delta"],-1.5)
        self.assertEqual(s["individual_matches"],3)
        with self.assertRaises(ValueError):summarize_trades(rows+rows[:1],0,60000)

    def test_negative_validation_blocks_positive_holdout(self):
        selected=dict(threshold=None,validation=dict(net_pnl=-1,trades=30))
        result=dict(net_pnl=100,trades=30,profit_factor=2,mean_daily_pnl_ci95=[1,3],realized_drawdown=.01)
        gates=checks_for(selected,result,result,{},result,P)
        self.assertFalse(gates["validation_positive"])
        self.assertFalse(all(gates.values()))


if __name__=="__main__":unittest.main()
