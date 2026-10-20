from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from PIL import Image
from g1_active_view.policies import OfflinePolicy
from g1_active_view.safety.low_head_recovery import LowHeadRecovery

def _image_validity(ref):
    try:
        if isinstance(ref, (str, Path)):
            with Image.open(ref) as im: arr=np.asarray(im.convert('RGB')).copy()
        else: arr=np.asarray(ref)
    except Exception as exc:
        return False, f'unreadable_image:{type(exc).__name__}'
    if arr.ndim != 3 or arr.shape[2] < 3: return False, f'expected_rgb_hwc_got:{arr.shape}'
    if not np.isfinite(arr).all(): return False, 'nonfinite_pixels'
    values=arr[:,:,:3].astype(np.float32)
    if float(values.mean()) <= .01 or float(values.std()) <= .01: return False, 'low_content_or_variance'
    return True, None

class OfflineRuntime:
    def __init__(self, model, recovery=None): self.model=model; self.recovery=recovery
    def reset(self):
        if self.recovery: self.recovery.reset()
    def predict(self, current_rgb, target_rgb, action_history=(), *, state=None, timestamp_s=None, pitch_step_deg=None):
        t0=time.perf_counter(); current_ok,current_reason=_image_validity(current_rgb); target_ok,target_reason=_image_validity(target_rgb)
        raw=self.model.predict(current_rgb,target_rgb,action_history); selected=raw['action_name']; audit={'override':False,'reason':'disabled'}
        if self.recovery is not None:
            if state is None or timestamp_s is None or pitch_step_deg is None or 'pitch_deg' not in state:
                audit={'override':False,'reason':'cannot_verify_missing_recovery_state'}
            else:
                selected,audit=self.recovery.choose(selected,pitch_deg=state['pitch_deg'],timestamp_s=timestamp_s,similarity=raw['similarity'],probabilities=raw['probabilities'],visual_inliers=raw['visual_inliers'],pitch_step_deg=pitch_step_deg,safe=bool(state.get('safe',False)))
        reasons=[x for x in (f'current:{current_reason}' if not current_ok and current_reason else None, f'target:{target_reason}' if not target_ok and target_reason else None) if x]
        return {'raw_action':raw['action_name'],'selected_action':selected,'protected_action':selected,'override':bool(audit.get('override',False)),'recovery':audit,'logits':raw['logits'],'probabilities':raw['probabilities'],'similarity':raw['similarity'],'visual_inliers':raw['visual_inliers'],'valid':not reasons,'failure_reason':';'.join(reasons) if reasons else None,'task_success':False,'task_success_reason':'offline_inference_only_stop_hold_not_verified','image_validity':{'current':current_ok,'target':target_ok},'latency_ms':(time.perf_counter()-t0)*1000}
