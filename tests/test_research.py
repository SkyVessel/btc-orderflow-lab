import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import valid_ohlc
from model import MLP
from zones import candidates, labels, FEATURES
from backtest import replay, recognition

CFG=json.loads((Path(__file__).resolve().parents[1]/"protocol.json").read_text())


def bars(n=78,date="2026-08-03",base=1785763800):
    rng=np.random.default_rng(8);c=100+np.cumsum(rng.normal(0,.5,n));o=np.r_[100,c[:-1]]
    return pd.DataFrame(dict(t=base+np.arange(n)*300,o=o,c=c,h=np.maximum(o,c)+.2,
        l=np.minimum(o,c)-.2,v=np.full(n,1e6),date=date,minute=570+np.arange(n)*5))


def ctx(b):
    return pd.DataFrame(0.,index=b.t,columns=["r1","r3","r6","d1","d3","d6","atrpct","logv","eff"])


def event(date,index,base,ident):
    return dict(id=ident,date=date,kind="orderblock",signal_t=base+(index+1)*300,
                signal_index=index,created_t=base+300,low=98.,high=99.,atr=1.,reference=100.,
                features=[0.]*len(FEATURES),probability=.9,predicted_returns=[.2]*3)


class Tests(unittest.TestCase):
    def test_no_future_in_candidates(self):
        b=bars();full,_=candidates(b,ctx(b),CFG)
        cut=b.iloc[:70];early,_=candidates(cut,ctx(cut),CFG)
        self.assertTrue(early)
        self.assertEqual(early,[r for r in full if r["signal_t"]<=cut.t.iloc[-1]+300])
        changed=b.copy();changed.loc[70:,["o","h","l","c"]]*=5
        altered,_=candidates(changed,ctx(changed),CFG)
        self.assertEqual(early,[r for r in altered if r["signal_t"]<=cut.t.iloc[-1]+300])

    def test_zones_known_before_touch(self):
        b=bars();events,register=candidates(b,ctx(b),CFG)
        for e in events:self.assertLess(e["created_t"],e["signal_t"])
        for z in register:
            if z["kind"]=="pressure":self.assertGreaterEqual(z["created_t"]-z["base_t"],900)

    def test_same_bar_ambiguity_is_failure(self):
        b=bars();base=int(b.t.iloc[0]);e=event(b.date.iloc[0],20,base,"a")
        b.loc[20,["o","h","l","c"]]=[99,99.5,98.5,99]
        b.loc[21,["o","h","l","c"]]=[99,101,96,99]
        rows,n=labels([e],b,CFG)
        self.assertEqual(n,0);self.assertEqual(rows[0]["label"],0)
        self.assertTrue(rows[0]["ambiguous"])

    def test_end_session_labels_excluded(self):
        b=bars();e=event(b.date.iloc[0],75,int(b.t.iloc[0]),"a")
        rows,n=labels([e],b,CFG);self.assertEqual((len(rows),n),(0,1))

    def cash_setup(self):
        b=bars();b[["o","h","l","c"]]=[100,110,99,110]
        c=copy.deepcopy(CFG);c.update(commission_bps_per_side=0,half_spread_bps=0,slippage_bps_per_side=0,allocation_fraction=1.)
        es=[event(b.date.iloc[0],i,int(b.t.iloc[0]),str(i)) for i in [20,24,28]]
        return b,c,es

    def test_no_same_day_reuse(self):
        b,c,es=self.cash_setup();r=replay(b,es,[b.date.iloc[0]],c,dict(hold_bars=1,threshold=.5))
        self.assertEqual(r["trades"],1);self.assertAlmostEqual(r["end_equity"],11000)

    def test_next_day_settlement_and_no_reset(self):
        b,c,es=self.cash_setup();b2=b.copy();b2.date="2026-08-04";b2.t+=86400
        e2=event("2026-08-04",20,int(b2.t.iloc[0]),"d2")
        r=replay(pd.concat([b,b2]),es+[e2],[b.date.iloc[0],"2026-08-04"],c,dict(hold_bars=1,threshold=.5))
        self.assertEqual(r["trades"],2);self.assertAlmostEqual(r["end_equity"],12100)
        self.assertEqual(r["daily"][1]["start_equity"],11000)

    def test_fees_reserved_and_accounting_identity(self):
        b,c,es=self.cash_setup();c.update(commission_bps_per_side=20,half_spread_bps=5,slippage_bps_per_side=5)
        r=replay(b,es,[b.date.iloc[0]],c,dict(hold_bars=1,threshold=.5))
        self.assertGreaterEqual(r["ledger"][0]["settled_remaining"],0)
        self.assertAlmostEqual(r["net_pnl"],r["gross_pnl"]-r["fees"]-r["spread_slippage"])

    def test_no_trade_day_counts(self):
        b,c,es=self.cash_setup();r=replay(b,[],[b.date.iloc[0]],c,dict(hold_bars=1,threshold=.5))
        self.assertEqual(r["daily_target_rate"],0);self.assertEqual(r["no_trade_days"],1)

    def test_pressure_does_not_short(self):
        b,c,es=self.cash_setup()
        for e in es:e["kind"]="pressure"
        r=replay(b,es,[b.date.iloc[0]],c,dict(hold_bars=1,threshold=.5))
        self.assertEqual(r["trades"],0)

    def test_future_volume_not_used_for_position(self):
        b,c,es=self.cash_setup();b.loc[21,"v"]=0
        r=replay(b,es[:1],[b.date.iloc[0]],c,dict(hold_bars=1,threshold=.5))
        self.assertEqual(r["ledger"][0]["shares"],100)

    def test_invalid_prices_rejected(self):
        b=bars();b.loc[2,"h"]=1
        with self.assertRaises(AssertionError):valid_ohlc(b)

    def test_mlp_learns_and_roundtrips(self):
        x=np.random.default_rng(4).normal(size=(150,3));y=(x[:,0]>0).astype(float)
        m=MLP(3,1,4,True);loss=m.fit(x,y,40,.01,32,.0001,4,True)
        self.assertLess(loss[-1],loss[0]);self.assertGreater(((m.predict(x).ravel()>.5)==y).mean(),.95)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"m.npz";m.save(p)
            np.testing.assert_array_equal(m.predict(x),MLP.load(p).predict(x))

    def test_recognition_counts_timeouts(self):
        rows=[dict(kind="pressure",probability=.8,label=0,outcome="timeout",ambiguous=False),
              dict(kind="pressure",probability=.8,label=1,outcome="success",ambiguous=False)]
        r=recognition(rows,[],.5,dict(pressure=.5,orderblock=.5))["pressure"]
        self.assertEqual(r["accepted_success_rate"],.5);self.assertEqual(r["timeout"],1)


if __name__=="__main__":unittest.main()
