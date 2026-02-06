import numpy as np

# ===============================
# 1) EMA + gap smoothing (как было)
# ===============================

def smooth_sequence(seq, vis_thr=0.35, max_gap=8, ema_alpha=0.25):
    """
    seq: (T,33,4) x,y,z,vis
    - держит последний валидный скелет max_gap кадров
    - EMA сглаживание координат
    """
    if seq is None or len(seq) == 0:
        return seq

    T, N, _ = seq.shape
    out = seq.copy()

    last_valid = None
    gap = 0

    for t in range(T):
        lm = out[t]
        vis = lm[:, 3]

        if np.nanmean(vis) >= vis_thr:
            if last_valid is not None:
                lm[:, :3] = (
                    ema_alpha * lm[:, :3]
                    + (1.0 - ema_alpha) * last_valid[:, :3]
                )
            last_valid = lm.copy()
            gap = 0
        else:
            if last_valid is not None and gap < max_gap:
                lm[:, :3] = last_valid[:, :3]
                lm[:, 3] = last_valid[:, 3]
                gap += 1
            else:
                gap = max_gap

        out[t] = lm

    return out


# ===============================
# 2) LR-flip stabilization (НОВОЕ)
# ===============================

LR_PAIRS = [
    (11, 12), (13, 14), (15, 16),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]

def _swap_lr_frame(lm):
    lm2 = lm.copy()
    for a, b in LR_PAIRS:
        lm2[[a, b]] = lm2[[b, a]]
    return lm2

def lock_lr_orientation(seq):
    """
    For back-view exercises (lat pulldown):
    fixes left/right once and never allows flipping.
    """
    if seq is None or len(seq) == 0:
        return seq

    out = seq.copy()

    # reference: shoulders X-order in first good frame
    ref = None
    for t in range(len(out)):
        l, r = out[t][11, 0], out[t][12, 0]
        if not np.isnan(l) and not np.isnan(r):
            ref = (l < r)
            break

    if ref is None:
        return out

    for t in range(len(out)):
        l, r = out[t][11, 0], out[t][12, 0]
        if np.isnan(l) or np.isnan(r):
            continue
        cur = (l < r)
        if cur != ref:
            out[t] = _swap_lr_frame(out[t])

    return out


def _lr_sign(lm):
    def pair(a, b):
        ax, bx = lm[a, 0], lm[b, 0]
        if np.isnan(ax) or np.isnan(bx):
            return None
        d = bx - ax
        if abs(d) < 1e-6:
            return None
        return 1 if d > 0 else -1

    s = pair(11, 12)
    if s is not None:
        return s
    return pair(23, 24)

def stabilize_lr_flip(seq, hysteresis_frames=3):
    """
    Убирает резкие перевороты скелета (left/right swap)
    """
    if seq is None or len(seq) == 0:
        return seq

    out = seq.copy()

    ref = None
    for t in range(len(out)):
        s = _lr_sign(out[t])
        if s is not None:
            ref = s
            break

    if ref is None:
        return out

    cnt = 0
    for t in range(len(out)):
        s = _lr_sign(out[t])
        if s is None:
            cnt = 0
            continue

        if s != ref:
            cnt += 1
            if cnt >= hysteresis_frames:
                out[t] = _swap_lr_frame(out[t])
        else:
            cnt = 0

    return out
