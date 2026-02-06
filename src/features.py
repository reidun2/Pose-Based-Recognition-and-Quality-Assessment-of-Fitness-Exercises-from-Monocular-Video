import numpy as np

def _angle(a, b, c):
    # angle at point b between ba and bc, returns radians
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosv)

def landmarks_to_features(seq):
    """
    seq: (T, 33, 4) -> use x,y only for angles
    output: (T, F)
    """
    xy = seq[:, :, :2]  # (T,33,2)

    # MediaPipe indices (common ones):
    # 11 left_shoulder, 12 right_shoulder
    # 23 left_hip, 24 right_hip
    # 25 left_knee, 26 right_knee
    # 27 left_ankle, 28 right_ankle
    # 13 left_elbow, 14 right_elbow
    # 15 left_wrist, 16 right_wrist

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16

    feats = []
    for t in range(xy.shape[0]):
        frame = xy[t]
        if np.isnan(frame).any():
            feats.append(np.full((8,), np.nan, dtype=np.float32))
            continue

        # knee angles
        l_knee = _angle(frame[L_HIP], frame[L_KNEE], frame[L_ANK])
        r_knee = _angle(frame[R_HIP], frame[R_KNEE], frame[R_ANK])

        # hip angles (shoulder-hip-knee as proxy for trunk/hip flexion)
        l_hip = _angle(frame[L_SH], frame[L_HIP], frame[L_KNEE])
        r_hip = _angle(frame[R_SH], frame[R_HIP], frame[R_KNEE])

        # elbow angles
        l_elb = _angle(frame[L_SH], frame[L_ELB], frame[L_WRI])
        r_elb = _angle(frame[R_SH], frame[R_ELB], frame[R_WRI])

        # trunk lean proxy: angle between shoulders-midhip-midankle (rough)
        mid_sh = (frame[L_SH] + frame[R_SH]) / 2
        mid_hip = (frame[L_HIP] + frame[R_HIP]) / 2
        mid_ank = (frame[L_ANK] + frame[R_ANK]) / 2
        trunk = _angle(mid_sh, mid_hip, mid_ank)

        # symmetry feature: abs diff between knees
        knee_diff = abs(l_knee - r_knee)

        feats.append(np.array([l_knee, r_knee, l_hip, r_hip, l_elb, r_elb, trunk, knee_diff], dtype=np.float32))

    return np.stack(feats, axis=0)  # (T, 8)

def pad_or_trim(x, T=120):
    # x: (t, f)
    t = x.shape[0]
    if t >= T:
        return x[:T]
    pad = np.repeat(x[-1:], T - t, axis=0) if t > 0 else np.zeros((T, x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)

def nan_to_num(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
