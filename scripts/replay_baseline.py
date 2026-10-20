#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from g1_active_view.policies import OfflinePolicy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--fixture',default='tests/fixtures/baseline.json'); p.add_argument('--output',default='tests/fixtures/deployment_output.json'); a=p.parse_args()
    f=json.loads((ROOT/a.fixture).read_text()); m=OfflinePolicy(ROOT/'models/va_sft_577_best.pt',ROOT/'third_party/dinov2',ROOT/'models/dinov2_vitb14_pretrain.pth','cpu'); rows=[]
    for r in f['cases']:
        out=m.predict(ROOT/r['current'],ROOT/r['target'],r['history']); rows.append({'id':r['id'],'output':out})
    (ROOT/a.output).write_text(json.dumps({'cases':rows},indent=2)+'\n'); print(json.dumps({'cases':rows},indent=2))
if __name__=='__main__': main()
