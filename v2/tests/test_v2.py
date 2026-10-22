import sys,json,copy,unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from regions import extract,region_labels,state_labels
from execution import plan,replay,activity

CFG=json.loads((HERE/"protocol.json").read_text())


def fixture():
    rng=np.random.default_rng(19);n=78;o=100+np.cumsum(rng.normal(0,.3,n));c=o+rng.normal(0,.3,n)
    return pd.DataFrame(dict(t=1785763800+np.arange(n)*300,o=o,c=c,h=np.maximum(o,c)+.3,
        l=np.minimum(o,c)-.3,v=100000,date="2026-08-03",minute=570+np.arange(n)*5))


def decision_rows(b,pred=-.1):
    return [dict(date=row.date,signal_t=int(row.t+300),signal_index=int(i),minute=int(row.minute+5),
        predicted_returns=[pred,pred,pred],snapshots={}) for i,row in b.iloc[:-6].iterrows()]


class V2Tests(unittest.TestCase):
    def test_regions_and_states_causal(self):
        b=fixture();e,z,s=extract(b,CFG);cut=b.iloc[:55];ee,zz,ss=extract(cut,CFG)
        self.assertTrue(ee);end=int(cut.t.iloc[-1]+300)
        self.assertEqual(ee,[r for r in e if r["signal_t"]<=end])
        self.assertEqual(ss,[r for r in s if r["signal_t"]<=end])
        altered=b.copy();altered.loc[55:,["o","h","l","c"]]*=5
        e2,_,s2=extract(altered,CFG)
        self.assertEqual(ee,[r for r in e2 if r["signal_t"]<=end])
        self.assertEqual(ss,[r for r in s2 if r["signal_t"]<=end])

    def test_multi_touch_unique_and_cooldown(self):
        e,z,s=extract(fixture(),CFG);self.assertTrue(e)
        for zone in z:
            q=[r for r in e if r["zone_id"]==zone["id"]]
            self.assertLessEqual(len(q),3)
            self.assertEqual([r["touch_number"] for r in q],list(range(1,len(q)+1)))
            for left,right in zip(q,q[1:]):self.assertGreaterEqual(right["signal_index"]-left["signal_index"],3)
            for r in q:self.assertLess(r["created_t"],r["signal_t"])

    def test_dense_labels_and_session_boundary(self):
        b=fixture();_,_,s=extract(b,CFG);r=state_labels(s,b,CFG)
        self.assertEqual(len(r),60)
        self.assertTrue(all(x["label_end_t"]<=b.t.iloc[-1]+300 for x in r))
        e=r[0];i=e["signal_index"]+1
        self.assertAlmostEqual(e["gross_returns"][2],b.c.iloc[i+5]/b.o.iloc[i]-1)

    def test_deadline_exploration_is_explicit(self):
        b=fixture();orders=plan(decision_rows(b),[b.date.iloc[0]],CFG,threshold=1)
        self.assertEqual(len(orders),3);self.assertTrue(all(o["forced_exploration"] for o in orders))
        for o,window in zip(orders,CFG["windows_minutes_ny"]):
            self.assertEqual(int(b.minute.iloc[o["signal_index"]]+5),window[1])

    def test_first_pass_not_future_best(self):
        b=fixture();rows=decision_rows(b,pred=.1)
        orders=plan(rows,[b.date.iloc[0]],CFG,threshold=0)
        self.assertEqual(len(orders),3)
        self.assertFalse(any(o["forced_exploration"] for o in orders))
        for o,w in zip(orders,CFG["windows_minutes_ny"]):
            self.assertEqual(int(b.minute.iloc[o["signal_index"]]+5),w[0])
        for r in rows:
            if r["minute"]==CFG["windows_minutes_ny"][0][1]:r["predicted_returns"]=[10,10,10]
        self.assertEqual(orders[0],plan(rows,[b.date.iloc[0]],CFG,0)[0])

    def test_cash_account_and_costs(self):
        b=fixture();orders=plan(decision_rows(b),[b.date.iloc[0]],CFG,1)
        r=replay(b,orders,[b.date.iloc[0]],CFG)
        self.assertEqual(r["trades"],3)
        self.assertLessEqual(r["daily"][0]["spent_settled_cash"],10000)
        self.assertAlmostEqual(r["net_pnl"],r["gross_pnl"]-r["fees"]-r["friction"])
        self.assertTrue(all(t["settled_remaining"]>=0 for t in r["ledger"]))

    def test_stress_replays_same_actions(self):
        b=fixture();orders=plan(decision_rows(b),[b.date.iloc[0]],CFG,1)
        a=replay(b,orders,[b.date.iloc[0]],CFG);s=replay(b,orders,[b.date.iloc[0]],CFG,2)
        self.assertEqual([(t["entry_t"],t["exit_t"]) for t in a["ledger"]],[(t["entry_t"],t["exit_t"]) for t in s["ledger"]])
        self.assertGreater(s["fees"]+s["friction"],a["fees"]+a["friction"])

    def test_zero_and_sparse_trading_fail(self):
        self.assertFalse(activity(dict(trades=0,days=12,no_trade_days=12),CFG)["passed"])
        self.assertFalse(activity(dict(trades=23,days=12,no_trade_days=0),CFG)["passed"])
        self.assertFalse(activity(dict(trades=30,days=12,no_trade_days=3),CFG)["passed"])
        self.assertTrue(activity(dict(trades=24,days=12,no_trade_days=2),CFG)["passed"])

    def test_no_reset_next_day(self):
        b=fixture();b[["o","h","l","c"]]=[100,102,99,102]
        b2=b.copy();b2.t+=86400;b2.date="2026-08-04"
        both=pd.concat([b,b2],ignore_index=True)
        rows=decision_rows(b)+decision_rows(b2)
        orders=plan(rows,["2026-08-03","2026-08-04"],CFG,1)
        r=replay(both,orders,["2026-08-03","2026-08-04"],CFG)
        self.assertGreater(r["daily"][1]["start_equity"],10000)
        self.assertEqual(r["daily"][0]["end_equity"],r["daily"][1]["start_equity"])


if __name__=="__main__":unittest.main()
