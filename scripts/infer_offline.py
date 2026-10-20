#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from g1_active_view.policies import OfflinePolicy
from g1_active_view.runtime import OfflineRuntime
from g1_active_view.safety import LowHeadRecovery

def main():
    p=argparse.ArgumentParser(description='Offline visual action inference; no camera or actuator access')
    p.add_argument('--current',required=True); p.add_argument('--target',required=True); p.add_argument('--history',default='[]'); p.add_argument('--device',default='cpu'); p.add_argument('--checkpoint',default=os.environ.get('G1_ACTIVE_VIEW_CHECKPOINT')); p.add_argument('--dino-source',default=os.environ.get('G1_ACTIVE_VIEW_DINO_SOURCE')); p.add_argument('--dino-weights',default=os.environ.get('G1_ACTIVE_VIEW_DINO_WEIGHTS')); p.add_argument('--recovery',action='store_true'); p.add_argument('--pitch-deg',type=float); p.add_argument('--timestamp-s',type=float); p.add_argument('--pitch-step-deg',type=float); p.add_argument('--safe',action='store_true'); p.add_argument('--output',type=Path)
    a=p.parse_args(); hist=json.loads(a.history)
    if not isinstance(hist,list) or len(hist)>8: raise SystemExit('--history must be a JSON list of at most 8 action names')
    model=OfflinePolicy(Path(a.checkpoint) if a.checkpoint else ROOT/'models/va_sft_577_best.pt', Path(a.dino_source) if a.dino_source else ROOT/'third_party/dinov2', Path(a.dino_weights) if a.dino_weights else ROOT/'models/dinov2_vitb14_pretrain.pth', a.device)
    recovery=LowHeadRecovery() if a.recovery else None; rt=OfflineRuntime(model,recovery)
    state={'pitch_deg':a.pitch_deg,'safe':a.safe} if a.pitch_deg is not None else None
    result=rt.predict(a.current,a.target,hist,state=state,timestamp_s=a.timestamp_s,pitch_step_deg=a.pitch_step_deg)
    result['inputs']={'current':str(Path(a.current).resolve()),'target':str(Path(a.target).resolve()),'history':hist}
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if a.output: a.output.write_text(text+'\n')
    print(text)
if __name__=='__main__': main()
