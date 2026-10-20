import json, math
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def _engine():
    from g1_active_view.policies import OfflinePolicy
    return OfflinePolicy(ROOT/'models/va_sft_577_best.pt', ROOT/'third_party/dinov2', ROOT/'models/dinov2_vitb14_pretrain.pth', 'cpu')

def test_replay_matches_original():
    from g1_active_view.actions import ACTIONS
    m=_engine(); baseline=json.loads((ROOT/'tests/fixtures/original_output.json').read_text())
    spec=json.loads((ROOT/'tests/fixtures/baseline.json').read_text())
    for case, expected in zip(spec['cases'], baseline['cases']):
        out=m.predict(ROOT/case['current'],ROOT/case['target'],case['history']); ref=expected['output']
        assert out['action_name']==ref['action_name']; assert out['action_id']==ref['action_id']; assert out['visual_inliers']==ref['visual_inliers']
        assert math.isclose(out['similarity'],ref['similarity'],abs_tol=1e-6)
        assert max(abs(a-b) for a,b in zip(out['logits'],ref['logits'])) <= 1e-6
        assert max(abs(a-b) for a,b in zip(out['probabilities'],ref['probabilities'])) <= 1e-6
        assert out['action_name'] in ACTIONS

def test_history_contract_and_reset():
    from g1_active_view.actions import encode_history, PAD_ID
    assert encode_history([])==[PAD_ID]*8
    assert encode_history(['MoveXPos']+['Stop']*8)==[8]*7+[8]
    with pytest.raises(ValueError): encode_history(['invalid'])

def test_recovery_requires_real_state_and_can_override():
    from g1_active_view.runtime import OfflineRuntime
    from g1_active_view.safety import LowHeadRecovery
    class Fake:
        def predict(self,*a): return {'raw_action':'MoveXPos','action_name':'MoveXPos','probabilities':[1/9]*9,'logits':[0.]*9,'similarity':.5,'visual_inliers':0}
    rt=OfflineRuntime(Fake(),LowHeadRecovery())
    r=rt.predict('x','y',[],state=None,timestamp_s=None,pitch_step_deg=None)
    assert r['override'] is False and r['recovery']['reason']=='cannot_verify_missing_recovery_state'
    rt.reset(); out=None
    for t in (0.,1.5,3.1): out=rt.predict('x','y',[],state={'pitch_deg':20.,'safe':True},timestamp_s=t,pitch_step_deg=5.)
    assert out['override'] is True and out['selected_action']=='GimbalPitchUp'

def test_stop_hold_missing_signals_never_pass():
    from g1_active_view.safety import verify_stop_hold
    r=verify_stop_hold(similarity=1.,threshold=.8,duration_s=3.,samples=[])
    assert r['verified'] is False
