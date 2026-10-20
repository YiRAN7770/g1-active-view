#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys
ROOT=Path(__file__).resolve().parents[1]
print('python',sys.version)
for name in ('torch','numpy','PIL','cv2'):
    spec=importlib.util.find_spec(name); print(name, 'available' if spec else 'MISSING')
for p in [ROOT/'models/va_sft_577_best.pt',ROOT/'models/dinov2_vitb14_pretrain.pth',ROOT/'third_party/dinov2']:
    print(str(p.relative_to(ROOT)), 'OK' if p.exists() else 'MISSING')
