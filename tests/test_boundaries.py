import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
ROOT=Path(__file__).resolve().parents[1]

def _engine():
    from g1_active_view.policies import OfflinePolicy
    return OfflinePolicy(ROOT/'models/va_sft_577_best.pt', ROOT/'third_party/dinov2', ROOT/'models/dinov2_vitb14_pretrain.pth', 'cpu')

def test_textureless_different_images_keep_raw_policy_but_no_success(tmp_path):
    from g1_active_view.runtime import OfflineRuntime
    a=tmp_path/'black.png'; b=tmp_path/'gray.png'; Image.new('RGB',(640,480),(0,0,0)).save(a); Image.new('RGB',(640,480),(128,128,128)).save(b)
    r=OfflineRuntime(_engine()).predict(a,b,[])
    assert r['raw_action'] in ('GimbalPitchUp','Stop','MoveXPos','MoveXNeg','MoveYPos','MoveYNeg','BodyYawLeft','BodyYawRight','GimbalPitchDown')
    assert r['visual_inliers']==0 and r['task_success'] is False and r['valid'] is False
    assert r['failure_reason'] is not None

def test_invalid_input_fails_before_policy(tmp_path):
    from g1_active_view.runtime import OfflineRuntime
    missing=tmp_path/'missing.png'
    with pytest.raises((ValueError,FileNotFoundError)):
        OfflineRuntime(_engine()).predict(missing,missing,[])

def test_stop_is_not_task_success_without_hold():
    from g1_active_view.runtime import OfflineRuntime
    p=ROOT/'datasets/fixtures/blank.png'
    r=OfflineRuntime(_engine()).predict(p,p,[])
    assert r['raw_action']=='Stop'; assert r['task_success'] is False; assert 'not_verified' in r['task_success_reason']

def test_recovery_stale_timestamp_is_rejected():
    from g1_active_view.runtime import OfflineRuntime
    from g1_active_view.safety import LowHeadRecovery
    class Fake:
        def predict(self,*args): return {'raw_action':'MoveXPos','action_name':'MoveXPos','probabilities':[1/9]*9,'logits':[0.]*9,'similarity':.5,'visual_inliers':0}
    rt=OfflineRuntime(Fake(),LowHeadRecovery()); state={'pitch_deg':20.,'safe':True}
    r1=rt.predict(np.ones((2,2,3),dtype=np.uint8)*80,np.ones((2,2,3),dtype=np.uint8)*80,[],state=state,timestamp_s=1.,pitch_step_deg=5.)
    r2=rt.predict(np.ones((2,2,3),dtype=np.uint8)*80,np.ones((2,2,3),dtype=np.uint8)*80,[],state=state,timestamp_s=1.,pitch_step_deg=5.)
    assert r1['recovery']['reason'] in ('waiting_for_persistence','conditions_not_met')
    assert r2['recovery']['reason']=='invalid_or_stale_observation'
