"""Pure visual low-head recovery guard, with no hardware access."""
from collections import deque
from dataclasses import dataclass, asdict
import math
@dataclass(frozen=True)
class RecoveryConfig:
    pitch_threshold_deg: float=5.0; low_similarity: float=.60; consecutive_observations: int=3; min_duration_s: float=3.; max_progress: float=.02; confidence_threshold: float=.45; margin_threshold: float=.15; min_visual_inliers: int=12; cooldown_decisions: int=3; max_overrides_per_episode: int=3
    def __post_init__(self):
        if not (self.pitch_threshold_deg>=0 and -1<=self.low_similarity<=1 and self.consecutive_observations>=2 and self.min_duration_s>0 and self.max_progress>=0 and 0<=self.confidence_threshold<=1 and 0<=self.margin_threshold<=1 and self.min_visual_inliers>=0 and self.cooldown_decisions>=1 and self.max_overrides_per_episode>=1): raise ValueError('invalid recovery configuration')
class LowHeadRecovery:
    def __init__(self, config=None): self.cfg=RecoveryConfig(**(config or {})); self.window=deque(maxlen=self.cfg.consecutive_observations); self.overrides=0; self.cooldown=0; self.last_time=None
    def reset(self): self.window.clear(); self.overrides=0; self.cooldown=0; self.last_time=None
    def choose(self, action, *, pitch_deg, timestamp_s, similarity, probabilities, visual_inliers, pitch_step_deg, safe=True):
        audit=dict(raw_action=action,selected_action=action,override=False,pitch_deg=pitch_deg,timestamp_s=timestamp_s,similarity=similarity,visual_inliers=visual_inliers,overrides_used=self.overrides)
        values=[pitch_deg,timestamp_s,similarity,visual_inliers,pitch_step_deg,*probabilities]; valid=all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) and len(probabilities)==9 and all(0<=v<=1 for v in probabilities) and abs(sum(probabilities)-1)<1e-3 and -1<=similarity<=1 and pitch_step_deg>0 and visual_inliers>=0
        if not valid or (self.last_time is not None and timestamp_s<=self.last_time): self.window.clear(); audit['reason']='invalid_or_stale_observation'; return action,audit
        self.last_time=timestamp_s; p=sorted(probabilities,reverse=True); uncertain=p[0]<self.cfg.confidence_threshold or p[0]-p[1]<self.cfg.margin_threshold or visual_inliers<self.cfg.min_visual_inliers; eligible=safe and pitch_deg>self.cfg.pitch_threshold_deg and similarity<self.cfg.low_similarity and uncertain; audit.update(uncertain=uncertain,top_probability=p[0],margin=p[0]-p[1])
        if self.cooldown: self.cooldown-=1; self.window.clear(); audit['reason']='cooldown'; return action,audit
        if not eligible: self.window.clear(); audit['reason']='conditions_not_met'; return action,audit
        self.window.append((timestamp_s,similarity)); duration=self.window[-1][0]-self.window[0][0]; progress=max(s for _,s in self.window)-self.window[0][1]; audit.update(consecutive_observations=len(self.window),duration_s=duration,progress=progress)
        if len(self.window)<self.cfg.consecutive_observations or duration<self.cfg.min_duration_s: audit['reason']='waiting_for_persistence'
        elif progress>self.cfg.max_progress: audit['reason']='visual_progress_present'
        elif self.overrides>=self.cfg.max_overrides_per_episode: audit['reason']='episode_budget_exhausted'
        elif pitch_deg-pitch_step_deg<0: audit['reason']='would_cross_horizontal'
        elif action=='GimbalPitchUp': audit['reason']='policy_already_pitching_up'; self.window.clear()
        else: self.overrides+=1; self.cooldown=self.cfg.cooldown_decisions; self.window.clear(); audit.update(selected_action='GimbalPitchUp',override=True,overrides_used=self.overrides,reason='persistent_low_head_visual_uncertainty')
        return audit['selected_action'],audit
