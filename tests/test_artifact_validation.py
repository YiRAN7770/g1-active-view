import hashlib, json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def _load_validator():
    import importlib.util
    spec=importlib.util.spec_from_file_location('prepare_models',ROOT/'scripts/prepare_models.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def _fixture_root(tmp_path, policy=b'policy', dino=b'dino'):
    (tmp_path/'models').mkdir(exist_ok=True); (tmp_path/'models/va_sft_577_best.pt').write_bytes(policy); (tmp_path/'models/dinov2_vitb14_pretrain.pth').write_bytes(dino)
    h=lambda x:hashlib.sha256(x).hexdigest()
    (tmp_path/'models/manifest.json').write_text(json.dumps({'policy_checkpoint':{'sha256':h(policy),'bytes':len(policy)},'dinov2':{'weights_sha256':h(dino),'weights_bytes':len(dino)}})); return tmp_path

def test_artifact_validator_accepts_exact_files(tmp_path, capsys):
    _load_validator().check_artifacts(_fixture_root(tmp_path)); assert 'MODEL_CHECK_PASS' in capsys.readouterr().out

def test_artifact_validator_rejects_missing_hash_and_lfs_pointer(tmp_path):
    mod=_load_validator(); root=_fixture_root(tmp_path); (root/'models/va_sft_577_best.pt').unlink()
    with pytest.raises(SystemExit,match='MISSING'): mod.check_artifacts(root)
    root=_fixture_root(tmp_path); (root/'models/va_sft_577_best.pt').write_bytes(b'changed')
    with pytest.raises(SystemExit,match='CHECKSUM_MISMATCH'): mod.check_artifacts(root)
    root=_fixture_root(tmp_path); (root/'models/va_sft_577_best.pt').write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 4\n')
    with pytest.raises(SystemExit,match='LFS pointer'): mod.check_artifacts(root)
