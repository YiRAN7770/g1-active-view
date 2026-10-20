#!/usr/bin/env python3
"""Verify local model artifacts; never download."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]

def _manifest(root): return json.loads((root/'models/manifest.json').read_text())
def _check(path, expected, expected_bytes):
    if not path.exists(): raise SystemExit(f'MISSING: {path}')
    if path.read_bytes()[:80].startswith(b'version https://git-lfs.github.com/spec'): raise SystemExit(f'LFS pointer, fetch required: {path}')
    size=path.stat().st_size; digest=hashlib.sha256(path.read_bytes()).hexdigest()
    print(path, size, digest)
    if size != expected_bytes or digest != expected: raise SystemExit(f'CHECKSUM_MISMATCH: {path}')

def check_artifacts(root=ROOT):
    m=_manifest(Path(root)); check=Path(root)/'models'
    _check(check/'va_sft_577_best.pt',m['policy_checkpoint']['sha256'],m['policy_checkpoint']['bytes'])
    _check(check/'dinov2_vitb14_pretrain.pth',m['dinov2']['weights_sha256'],m['dinov2']['weights_bytes'])
    print('MODEL_CHECK_PASS')

def main(): check_artifacts()
if __name__=='__main__': main()
