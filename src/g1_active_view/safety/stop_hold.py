"""Hardware-neutral Stop verifier.

The verifier consumes measured signals supplied by an adapter. Missing signals
never pass and no zero defaults are inserted.
"""
def verify_stop_hold(*, similarity, threshold, duration_s, samples,
                      linear_speed_max_m_s=0.025, yaw_rate_max_rad_s=0.04,
                      xy_drift_max_m=0.025, yaw_drift_max_deg=1.0,
                      pitch_drift_max_deg=0.25):
    if not samples:
        return {'verified':False,'reason':'missing_stability_samples'}
    required=('linear_speed_m_s','yaw_rate_rad_s','xy_drift_m','yaw_drift_deg','pitch_drift_deg')
    missing=[k for k in required if any(k not in s or s[k] is None for s in samples)]
    if missing: return {'verified':False,'reason':'missing_physical_signals','missing':missing}
    if similarity < threshold: return {'verified':False,'reason':'similarity_below_threshold'}
    if duration_s < 3.0: return {'verified':False,'reason':'hold_duration_below_3s'}
    limits={'linear_speed_m_s':linear_speed_max_m_s,'yaw_rate_rad_s':yaw_rate_max_rad_s,'xy_drift_m':xy_drift_max_m,'yaw_drift_deg':yaw_drift_max_deg,'pitch_drift_deg':pitch_drift_max_deg}
    violations=[k for k,limit in limits.items() if any(abs(float(s[k]))>limit for s in samples)]
    if violations: return {'verified':False,'reason':'stability_limit_exceeded','violations':violations}
    return {'verified':True,'reason':'measured_stable_hold','duration_s':float(duration_s)}
