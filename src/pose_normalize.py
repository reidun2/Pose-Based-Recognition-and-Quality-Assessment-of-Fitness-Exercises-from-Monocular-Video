import numpy as np

# MediaPipe indices
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24

def _safe_mean(a, b):
    if np.isnan(a).any() and np.isnan(b).any():
        return None
    if np.isnan(a).any(): 
        return b
    if np.isnan(b).any():
        return a
    return 0.5 * (a + b)

def normalize_frame(lm, eps=1e-6, rotate=True):
    """
    lm: (33,4) [x,y,z,vis] with x,y normalized to frame (0..1), z is relative.
    returns lm_norm: (33,4) normalized.
    """
    lm = lm.copy()

    # If the whole frame is missing
    if np.isnan(lm[:, 0]).all() or np.isnan(lm[:, 1]).all():
        return lm

    xy = lm[:, :2]
    z = lm[:, 2:3]
    v = lm[:, 3:4]

    # center at mid-hip
    hip_c = _safe_mean(xy[L_HIP], xy[R_HIP])
    if hip_c is None:
        # fallback: mid-shoulder if hips missing
        sh_c = _safe_mean(xy[L_SH], xy[R_SH])
        if sh_c is None:
            return lm
        center = sh_c
    else:
        center = hip_c

    xy = xy - center  # translation

    # scale by body size (prefer shoulder width, fallback hip width)
    scale = None
    if not np.isnan(xy[[L_SH, R_SH]]).any():
        scale = np.linalg.norm(xy[L_SH] - xy[R_SH])
    if (scale is None) or (scale < eps):
        if not np.isnan(xy[[L_HIP, R_HIP]]).any():
            scale = np.linalg.norm(xy[L_HIP] - xy[R_HIP])
    if (scale is None) or (scale < eps):
        # last fallback: max distance between any two visible points
        ok = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])
        pts = xy[ok]
        if len(pts) >= 2:
            # rough scale: max norm to origin
            scale = np.max(np.linalg.norm(pts, axis=1))
        else:
            scale = 1.0

    scale = max(float(scale), eps)
    xy = xy / scale
    z = z / scale  # keep z consistent with xy scale

    # rotate so shoulders are horizontal
    if rotate and (not np.isnan(xy[[L_SH, R_SH]]).any()):
        vec = xy[R_SH] - xy[L_SH]
        angle = np.arctan2(vec[1], vec[0])  # radians
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        xy = (xy @ R.T)

    lm[:, :2] = xy
    lm[:, 2:3] = z
    lm[:, 3:4] = v
    return lm

def normalize_sequence(seq, rotate=True):
    """
    seq: (T,33,4)
    """
    out = np.empty_like(seq, dtype=np.float32)
    for t in range(seq.shape[0]):
        out[t] = normalize_frame(seq[t], rotate=rotate)
    return out
