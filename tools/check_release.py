"""Offline integrity and inference smoke check; this does not prove profitability."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data import sha
from model import MLP


def main():
    out=ROOT/'results/v0.2'
    for file,digest in json.loads((out/'artifact_hashes.json').read_text()).items():
        if sha(out/file)!=digest:raise ValueError('Artifact changed: '+file)
    for file,digest in json.loads((out/'source_hashes.json').read_text()).items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen research source changed: '+file)
    for file,n,width in [('scratch_region.npz',17,1),('scratch_returns.npz',35,3)]:
        model=MLP.load(out/file);p=model.predict(np.zeros((1,n)))
        if p.shape!=(1,width) or not np.isfinite(p).all():raise ValueError('Inference failed')
    print('PASS: original v0.2 artifacts/source hashes and selected model inference. Profit qualification: FAILED. Cloud runtime: NOT STARTED.')


if __name__=='__main__':main()
