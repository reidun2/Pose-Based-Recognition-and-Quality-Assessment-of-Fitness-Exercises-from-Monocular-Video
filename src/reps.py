# reps.py
import numpy as np
from extract_signals import get_primary_signal, lateral_raise_depth_with_fallback  # единый источник "primary signal" для squat/bench/и т.д.


# ----------------- small utils -----------------

def angle_3pts(a, b, c, eps=1e-9):
    """Angle ABC in degrees for 2D points a,b,c (each: (x,y))."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + eps
    nbc = np.linalg.norm(bc) + eps
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _fill_nans_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    for i in range(len(x) - 2, -1, -1):
        if np.isnan(x[i]):
            x[i] = x[i + 1]
    return x


def smooth_1d(x, win=11):
    x = _fill_nans_1d(np.asarray(x, dtype=np.float32))
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")


# ----------------- signals used by LAT PULLDOWN -----------------

def elbow_angle_signal(seq, side="left"):
    # MediaPipe: L shoulder/elbow/wrist = 11/13/15, R = 12/14/16
    if side == "left":
        sh, el, wr = 11, 13, 15
    else:
        sh, el, wr = 12, 14, 16

    T = seq.shape[0]
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        if np.isnan(lm[[sh, el, wr], :2]).any():
            continue
        a = lm[sh, :2]
        b = lm[el, :2]
        c = lm[wr, :2]
        sig[t] = angle_3pts(a, b, c)
    return sig


def elbow_angle_signal_both(seq):
    sigL = elbow_angle_signal(seq, "left")
    sigR = elbow_angle_signal(seq, "right")
    return np.nanmean(np.stack([sigL, sigR], axis=0), axis=0).astype(np.float32)


def elbows_below_shoulders_signal(seq, margin=0.02):
    """
    gate[t] = 1 if mean(elbows_y) > mean(shoulders_y) + margin else 0
    (y растёт вниз)
    """
    T = seq.shape[0]
    out = np.zeros((T,), dtype=np.float32)

    for t in range(T):
        lm = seq[t]

        ys = []
        ye = []

        for sh in (11, 12):
            y = lm[sh, 1]
            if not np.isnan(y):
                ys.append(float(y))

        for el in (13, 14):
            y = lm[el, 1]
            if not np.isnan(y):
                ye.append(float(y))

        if not ys or not ye:
            out[t] = 0.0
            continue

        shoulder_y = float(np.mean(ys))
        elbow_y = float(np.mean(ye))
        out[t] = 1.0 if (elbow_y > shoulder_y + margin) else 0.0

    return out


# ----------------- rep segmentation helpers -----------------

def find_local_minima(sig, min_dist=15, prominence=6.0):
    sig = np.asarray(sig, dtype=np.float32)
    mins = []
    for i in range(1, len(sig) - 1):
        if sig[i] < sig[i - 1] and sig[i] < sig[i + 1]:
            mins.append(i)

    good = []
    for i in mins:
        left = max(0, i - min_dist)
        right = min(len(sig) - 1, i + min_dist)
        local_max = max(np.max(sig[left:i + 1]), np.max(sig[i:right + 1]))
        if (local_max - sig[i]) >= prominence:
            good.append(i)

    # deepest first, then enforce spacing
    good = sorted(good, key=lambda t: sig[t])
    selected = []
    for i in good:
        if all(abs(i - j) >= min_dist for j in selected):
            selected.append(i)

    return sorted(selected)


def dedup_bottoms(bottoms, sig, min_sep=12):
    """
    If two bottoms are closer than min_sep -> keep the deeper one (smaller sig).
    """
    if not bottoms:
        return []
    sig = np.asarray(sig, dtype=np.float32)
    bottoms = sorted([int(b) for b in bottoms])

    kept = [bottoms[0]]
    for b in bottoms[1:]:
        if (b - kept[-1]) < min_sep:
            if sig[b] < sig[kept[-1]]:
                kept[-1] = b
        else:
            kept.append(b)
    return kept


def find_tops_between_bottoms(sig, bottoms):
    """
    For each gap bottoms[i]..bottoms[i+1], find a top as max(sig) inside the middle 80%.
    """
    sig = np.asarray(sig, dtype=np.float32)
    tops = []
    for i in range(len(bottoms) - 1):
        a, b = bottoms[i], bottoms[i + 1]
        if b <= a + 1:
            tops.append(a)
            continue
        lo = a + (b - a) // 10
        hi = b - (b - a) // 10
        j = int(np.argmax(sig[lo:hi + 1]) + lo)
        tops.append(j)
    return tops


def reps_from_bottoms_and_tops(bottoms, tops, T):
    """
    Creates reps: (prev_top -> bottom -> next_top)
    """
    if len(bottoms) == 0:
        return []

    reps = []
    if len(tops) > 0:
        reps.append((0, bottoms[0], tops[0]))
    else:
        reps.append((0, bottoms[0], T - 1))

    for i in range(1, len(bottoms) - 1):
        start = tops[i - 1] if (i - 1) < len(tops) else 0
        end = tops[i] if i < len(tops) else (T - 1)
        reps.append((start, bottoms[i], end))

    if len(bottoms) > 1:
        start = tops[-1] if len(tops) > 0 else 0
        reps.append((start, bottoms[-1], T - 1))

    clean = []
    for st, bt, en in reps:
        st = max(0, int(st))
        bt = max(0, int(bt))
        en = min(T - 1, int(en))
        if st < en and st <= bt <= en:
            clean.append((st, bt, en))
    return clean


def reps_by_hysteresis_lat(sigA, gate, T, bottom_thr, top_thr, min_gap=12, hold=2, gate_hold=2):
    """
    Lat pulldown:
      - сигнал: elbow angle (меньше = сильнее согнул локти/потянул вниз)
      - bottom: sigA <= bottom_thr
      - top:    sigA >= top_thr
      - gate: elbows below shoulders must be true at bottom (несколько кадров подряд)
    """
    reps = []
    state = "seek_bottom"
    last_switch = -10**9
    bottom_idx = None

    below_cnt = 0
    above_cnt = 0
    gate_cnt = 0

    for t in range(T):
        v = sigA[t]
        if np.isnan(v):
            below_cnt = above_cnt = gate_cnt = 0
            continue

        if state == "seek_bottom":
            below_cnt = below_cnt + 1 if v <= bottom_thr else 0
            gate_cnt = gate_cnt + 1 if gate[t] >= 0.5 else 0

            if below_cnt >= hold and gate_cnt >= gate_hold and (t - last_switch) >= min_gap:
                bottom_idx = t
                last_switch = t
                state = "seek_top"
                below_cnt = above_cnt = gate_cnt = 0

        else:  # seek_top
            above_cnt = above_cnt + 1 if v >= top_thr else 0
            if above_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                start = reps[-1][2] if reps else 0
                reps.append((start, bottom_idx, top_idx))
                last_switch = t
                state = "seek_bottom"
                below_cnt = above_cnt = gate_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def reps_by_hysteresis_lateral(sig, T, top_thr, down_thr, min_gap=8, hold=2, require_down_ready=True, top_is_low=True):
    """
    Lateral raises:
      - sig: arm depth relative to shoulders (lower = higher raise)
      - top (counted raise): sig <= top_thr  (arms at shoulder level or higher)
      - down: sig >= down_thr (arms lowered back)
    A rep is counted only after full top is reached and then returned down.
    """
    reps = []
    state = "seek_down_ready" if require_down_ready else "seek_top"
    top_idx = None
    last_switch = -10**9

    top_cnt = 0
    down_cnt = 0

    def is_top(v):
        return (v <= top_thr) if top_is_low else (v >= top_thr)

    def is_down(v):
        return (v >= down_thr) if top_is_low else (v <= down_thr)

    for t in range(T):
        v = sig[t]
        if np.isnan(v):
            top_cnt = 0
            down_cnt = 0
            continue

        if state == "seek_down_ready":
            down_cnt = down_cnt + 1 if is_down(v) else 0
            if down_cnt >= hold:
                state = "seek_top"
                top_cnt = 0
                down_cnt = 0
        elif state == "seek_top":
            top_cnt = top_cnt + 1 if is_top(v) else 0
            if top_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                last_switch = t
                state = "seek_down"
                top_cnt = 0
                down_cnt = 0
        else:
            down_cnt = down_cnt + 1 if is_down(v) else 0
            if down_cnt >= hold and (t - last_switch) >= min_gap:
                end_idx = t
                start_idx = reps[-1][2] if reps else 0
                reps.append((start_idx, top_idx, end_idx))
                last_switch = t
                # Next rep must re-confirm true bottom position first.
                state = "seek_down_ready" if require_down_ready else "seek_top"
                top_cnt = 0
                down_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def reps_by_extrema_lateral(sig, T, top_is_low=True, stride=1):
    """
    Alternate detector for lateral raises:
    build cycles from consecutive "down" extrema with one "top" between them.
    Helps split merged reps when hysteresis misses a down crossing.
    """
    sig = np.asarray(sig, dtype=np.float32)
    if not np.isfinite(sig).any():
        return []

    p10 = float(np.nanpercentile(sig, 10))
    p90 = float(np.nanpercentile(sig, 90))
    rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
    if rng < 1e-6:
        return []

    min_dist = max(6, int(12 / max(1, stride)))
    min_sep = max(4, int(0.6 * min_dist))
    prom = max(0.015, 0.14 * rng)

    if top_is_low:
        down_pts = find_local_minima(-sig, min_dist=min_dist, prominence=prom)
        down_pts = dedup_bottoms(down_pts, -sig, min_sep=min_sep)
        down_pts = [i for i in down_pts if sig[i] >= (p10 + 0.52 * rng)]
    else:
        down_pts = find_local_minima(sig, min_dist=min_dist, prominence=prom)
        down_pts = dedup_bottoms(down_pts, sig, min_sep=min_sep)
        down_pts = [i for i in down_pts if sig[i] <= (p90 - 0.52 * rng)]

    if len(down_pts) < 2:
        return []

    amp_min = max(0.03, 0.18 * rng)
    min_len = max(8, int(12 / max(1, stride)))
    reps = []

    for i in range(len(down_pts) - 1):
        st = int(down_pts[i])
        en = int(down_pts[i + 1])
        if en - st < min_len:
            continue

        lo = st + max(1, (en - st) // 8)
        hi = en - max(1, (en - st) // 8)
        if hi <= lo:
            continue

        if top_is_low:
            bt = int(np.argmin(sig[lo:hi + 1]) + lo)
            amp = 0.5 * (float(sig[st]) + float(sig[en])) - float(sig[bt])
        else:
            bt = int(np.argmax(sig[lo:hi + 1]) + lo)
            amp = float(sig[bt]) - 0.5 * (float(sig[st]) + float(sig[en]))

        if amp >= amp_min and st < bt < en:
            reps.append((st, bt, en))

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def lateral_abduction_signal(seq, vis_thr=0.35):
    """
    Shoulder abduction proxy in degrees (0..90):
    angle between torso vector (shoulder-hip) and upper arm (elbow/hand-shoulder),
    reduced to acute angle: min(theta, 180-theta).
    Down position ~ low angle, top position ~ high angle.
    """
    T = int(seq.shape[0])
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        vals = []
        # left side
        for sh, hp, el, wr in ((11, 23, 13, 15), (12, 24, 14, 16)):
            sh_xy = lm[sh, :2]
            hp_xy = lm[hp, :2]
            if np.isnan(sh_xy).any() or np.isnan(hp_xy).any():
                continue
            torso = sh_xy - hp_xy
            n_t = float(np.linalg.norm(torso))
            if n_t < 1e-6:
                continue

            arm_pt = None
            v_el = float(lm[el, 3]) if np.isfinite(lm[el, 3]) else 0.0
            v_wr = float(lm[wr, 3]) if np.isfinite(lm[wr, 3]) else 0.0
            if np.isfinite(lm[el, :2]).all() and v_el >= vis_thr:
                arm_pt = lm[el, :2]
            elif np.isfinite(lm[wr, :2]).all() and v_wr >= vis_thr:
                arm_pt = lm[wr, :2]
            if arm_pt is None:
                continue

            arm = arm_pt - sh_xy
            n_a = float(np.linalg.norm(arm))
            if n_a < 1e-6:
                continue
            cosang = float(np.clip(np.dot(torso, arm) / (n_t * n_a), -1.0, 1.0))
            theta = float(np.degrees(np.arccos(cosang)))
            theta = min(theta, 180.0 - theta)
            vals.append(theta)
        if vals:
            sig[t] = float(np.mean(vals))
    return sig


def reps_by_hysteresis_lateral_angle(sig, T, top_thr, down_thr, min_gap=8, hold=2):
    """
    Strict down->top->down counting using abduction angle signal.
    """
    reps = []
    state = "seek_down_ready"
    last_switch = -10**9
    top_idx = None
    top_cnt = 0
    down_cnt = 0

    for t in range(T):
        v = sig[t]
        if np.isnan(v):
            top_cnt = 0
            down_cnt = 0
            continue

        if state == "seek_down_ready":
            down_cnt = down_cnt + 1 if v <= down_thr else 0
            if down_cnt >= hold:
                state = "seek_top"
                top_cnt = 0
                down_cnt = 0
        elif state == "seek_top":
            top_cnt = top_cnt + 1 if v >= top_thr else 0
            if top_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                last_switch = t
                state = "seek_down"
                top_cnt = 0
                down_cnt = 0
        else:
            down_cnt = down_cnt + 1 if v <= down_thr else 0
            if down_cnt >= hold and (t - last_switch) >= min_gap:
                end_idx = t
                st_idx = reps[-1][2] if reps else 0
                reps.append((st_idx, top_idx, end_idx))
                last_switch = t
                state = "seek_top"
                top_cnt = 0
                down_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def reps_by_peaks_valleys_lateral_angle(sig, T, top_thr, min_gap=6, stride=1):
    """
    Fallback splitter: top peaks + valley after each peak.
    Useful when hysteresis merges adjacent reps.
    """
    sig = np.asarray(sig, dtype=np.float32)
    if not np.isfinite(sig).any():
        return []

    p10 = float(np.nanpercentile(sig, 10))
    p90 = float(np.nanpercentile(sig, 90))
    rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
    prom = max(2.5, 0.10 * rng)
    md = max(min_gap, int(8 / max(1, stride)))

    peaks = find_local_minima(-sig, min_dist=md, prominence=prom)
    peaks = dedup_bottoms(peaks, -sig, min_sep=max(3, int(0.6 * md)))
    peaks = [int(i) for i in peaks if np.isfinite(sig[i]) and sig[i] >= top_thr]
    if len(peaks) == 0:
        return []

    reps = []
    last_end = 0
    for i, pk in enumerate(peaks):
        lo = pk + 1
        if i + 1 < len(peaks):
            hi = max(lo, peaks[i + 1] - 1)
        else:
            hi = T - 1
        if hi <= lo or not np.isfinite(sig[lo:hi + 1]).any():
            continue
        en = int(np.argmin(sig[lo:hi + 1]) + lo)
        st = int(last_end)
        if st < en and st <= pk <= en:
            reps.append((st, int(pk), en))
            last_end = en
    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def _robust_norm_01(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.full_like(x, np.nan, dtype=np.float32)
    ok = np.isfinite(x)
    if not ok.any():
        return out
    p10 = float(np.nanpercentile(x, 10))
    p90 = float(np.nanpercentile(x, 90))
    rng = max(1e-6, p90 - p10)
    out[ok] = (x[ok] - p10) / rng
    out = np.clip(out, 0.0, 1.0)
    return out


def build_lateral_upscore(seq):
    """
    Build a generic up-score for lateral raises:
      higher score -> arms more raised.
    Combines:
      - arm abduction angle proxy
      - depth signal (wrist/elbow relative to shoulders)
    """
    ang = lateral_abduction_signal(seq, vis_thr=0.35)          # high at top
    dep = lateral_raise_depth_with_fallback(seq, vis_thr=0.35) # low at top

    ang_n = _robust_norm_01(ang)
    dep_n = _robust_norm_01(-dep)

    score = np.full_like(ang_n, np.nan, dtype=np.float32)
    for i in range(len(score)):
        a = ang_n[i]
        d = dep_n[i]
        if np.isfinite(a) and np.isfinite(d):
            score[i] = float(0.65 * a + 0.35 * d)
        elif np.isfinite(a):
            score[i] = float(a)
        elif np.isfinite(d):
            score[i] = float(d)
    return score


def reps_by_hysteresis_upscore(up, T, top_thr, down_thr, min_gap=6, hold=2):
    """
    Count strict down->top->down cycles using up-score (higher=arms up).
    """
    reps = []
    state = "seek_down_ready"
    top_idx = None
    last_switch = -10**9
    top_cnt = 0
    down_cnt = 0

    for t in range(T):
        v = up[t]
        if np.isnan(v):
            top_cnt = 0
            down_cnt = 0
            continue

        if state == "seek_down_ready":
            down_cnt = down_cnt + 1 if v <= down_thr else 0
            if down_cnt >= hold:
                state = "seek_top"
                top_cnt = 0
                down_cnt = 0
        elif state == "seek_top":
            top_cnt = top_cnt + 1 if v >= top_thr else 0
            if top_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                last_switch = t
                state = "seek_down"
                top_cnt = 0
                down_cnt = 0
        else:
            down_cnt = down_cnt + 1 if v <= down_thr else 0
            if down_cnt >= hold and (t - last_switch) >= min_gap:
                end_idx = t
                st_idx = reps[-1][2] if reps else 0
                reps.append((st_idx, top_idx, end_idx))
                last_switch = t
                state = "seek_top"
                top_cnt = 0
                down_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def _down_anchors_from_upscore(up, down_thr, stride=1):
    """
    Extract robust "down" anchors (arms lowered) from up-score.
    """
    up = np.asarray(up, dtype=np.float32)
    T = int(len(up))
    if T == 0 or not np.isfinite(up).any():
        return []

    anchors = []
    in_run = False
    run_st = 0
    for t in range(T):
        v = up[t]
        is_down = np.isfinite(v) and (v <= down_thr)
        if is_down and not in_run:
            in_run = True
            run_st = t
        elif (not is_down) and in_run:
            run_en = t - 1
            if run_en >= run_st and np.isfinite(up[run_st:run_en + 1]).any():
                j = int(np.argmin(up[run_st:run_en + 1]) + run_st)
                anchors.append(j)
            in_run = False
    if in_run:
        run_en = T - 1
        if run_en >= run_st and np.isfinite(up[run_st:run_en + 1]).any():
            j = int(np.argmin(up[run_st:run_en + 1]) + run_st)
            anchors.append(j)

    p10 = float(np.nanpercentile(up, 10))
    p90 = float(np.nanpercentile(up, 90))
    rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
    min_dist = max(3, int(6 / max(1, stride)))
    prom = max(0.012, 0.06 * max(rng, 1e-6))
    mins = find_local_minima(up, min_dist=min_dist, prominence=prom)
    mins = [int(i) for i in mins if np.isfinite(up[i]) and (up[i] <= (p10 + 0.52 * rng))]

    all_pts = sorted(set([int(i) for i in anchors] + mins))
    if len(all_pts) == 0:
        return []
    min_sep = max(3, int(5 / max(1, stride)))
    return dedup_bottoms(all_pts, up, min_sep=min_sep)


def _split_long_lateral_cycles(up, reps, top_thr, down_thr, stride=1):
    """
    Split very long cycles when they likely contain multiple reps merged together.
    """
    if len(reps) <= 1:
        return reps
    up = np.asarray(up, dtype=np.float32)
    lens = [int(en - st) for (st, _, en) in reps]
    med = float(np.median(lens)) if lens else 0.0
    long_thr = max(int(34 / max(1, stride)), int(1.8 * max(1.0, med)))
    min_piece = max(6, int(8 / max(1, stride)))

    out = []
    for (st, bt, en) in reps:
        st = int(st)
        bt = int(bt)
        en = int(en)
        if (en - st) < long_thr:
            out.append((st, bt, en))
            continue
        if not np.isfinite(up[st:en + 1]).any():
            out.append((st, bt, en))
            continue

        md = max(3, int(6 / max(1, stride)))
        prom_top = max(0.015, 0.08 * max(1e-6, float(np.nanpercentile(up, 90) - np.nanpercentile(up, 10))))
        seg = up[st:en + 1]
        peaks = find_local_minima(-seg, min_dist=md, prominence=prom_top)
        peaks = [int(st + p) for p in peaks if np.isfinite(up[int(st + p)]) and up[int(st + p)] >= top_thr]
        peaks = sorted(peaks)

        if len(peaks) < 2:
            out.append((st, bt, en))
            continue

        made_split = False
        for i in range(len(peaks) - 1):
            a = int(peaks[i])
            b = int(peaks[i + 1])
            if b - a < 2 * min_piece:
                continue
            cut = int(np.argmin(up[a:b + 1]) + a)
            if up[cut] > (down_thr + 0.10):
                continue
            if (cut - st) < min_piece or (en - cut) < min_piece:
                continue

            left_top = int(np.argmax(up[st:cut + 1]) + st)
            right_top = int(np.argmax(up[cut:en + 1]) + cut)
            if st < left_top < cut and cut < right_top < en:
                out.append((st, left_top, cut))
                out.append((cut, right_top, en))
                made_split = True
                break
        if not made_split:
            out.append((st, bt, en))

    chained = []
    for (st, bt, en) in out:
        st_i = int(chained[-1][2]) if chained else int(st)
        if st_i < int(en) and st_i <= int(bt) <= int(en):
            chained.append((st_i, int(bt), int(en)))
    return chained


def split_long_reps_by_local_down(up, reps, stride=1):
    if len(reps) <= 1:
        return reps
    lens = [int(en - st) for (st, _, en) in reps]
    med = float(np.median(lens)) if lens else 0.0
    long_thr = max(int(36 / max(1, stride)), int(1.7 * max(1.0, med)))
    min_piece = max(6, int(8 / max(1, stride)))
    out = []
    for (st, bt, en) in reps:
        if (en - st) < long_thr:
            out.append((int(st), int(bt), int(en)))
            continue
        seg = np.asarray(up[st:en + 1], dtype=np.float32)
        if not np.isfinite(seg).any():
            out.append((int(st), int(bt), int(en)))
            continue
        # split at strongest local down after top
        lo = int(bt) + 1
        hi = int(en) - 1
        if hi <= lo:
            out.append((int(st), int(bt), int(en)))
            continue
        cut = int(np.argmin(up[lo:hi + 1]) + lo)
        if (cut - st) >= min_piece and (en - cut) >= min_piece:
            # second top on right chunk
            r_lo = cut + 1
            if r_lo < en:
                r_top = int(np.argmax(up[r_lo:en + 1]) + r_lo)
                if st <= bt <= cut and cut <= r_top <= en:
                    out.append((int(st), int(bt), int(cut)))
                    out.append((int(cut), int(r_top), int(en)))
                    continue
        out.append((int(st), int(bt), int(en)))
    chained = []
    for (st, bt, en) in out:
        st_i = int(chained[-1][2]) if chained else int(st)
        if st_i < int(en) and st_i <= int(bt) <= int(en):
            chained.append((st_i, int(bt), int(en)))
    return chained


def reps_lateral_strict_cycles(sig, T, top_is_low=True, stride=1):
    """
    Strict lateral-raise segmentation: only down->top->down cycles.
    This guarantees rep end at the lowered position.
    """
    sig = np.asarray(sig, dtype=np.float32)
    if not np.isfinite(sig).any():
        return [], 0.0

    # Convert orientation so that in u-signal:
    # top = local maxima, down = local minima.
    u = (-sig) if top_is_low else sig
    if not np.isfinite(u).any():
        return [], 0.0

    p10 = float(np.nanpercentile(u, 10))
    p90 = float(np.nanpercentile(u, 90))
    rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
    if rng < 1e-6:
        return [], 0.0

    # More permissive cycle extraction to avoid merging neighboring reps.
    min_dist = max(3, int(6 / max(1, stride)))
    min_sep = max(2, int(0.5 * min_dist))
    prom = max(0.006, 0.04 * rng)
    min_len = max(6, int(8 / max(1, stride)))
    amp_thr = max(0.018, 0.08 * rng)

    downs = find_local_minima(u, min_dist=min_dist, prominence=prom)
    downs = dedup_bottoms(downs, u, min_sep=min_sep)
    if len(downs) < 2:
        return [], 0.0

    reps = []
    amps = []
    for i in range(len(downs) - 1):
        st = int(downs[i])
        en = int(downs[i + 1])
        if en - st < min_len:
            continue
        a = st + max(1, (en - st) // 10)
        b = en - max(1, (en - st) // 10)
        if b <= a:
            continue
        top = int(np.argmax(u[a:b + 1]) + a)
        if not (st < top < en):
            continue
        amp = min(float(u[top] - u[st]), float(u[top] - u[en]))
        if amp < amp_thr:
            continue
        reps.append((st, top, en))
        amps.append(amp)

    score = float(np.mean(amps)) if amps else 0.0
    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en], score


def refine_lateral_rep_ends_to_down(sig, reps, T, top_is_low=True, stride=1):
    """
    Force each rep end to a down extremum after top and before next top.
    This prevents endings while the arm is still raised.
    """
    if len(reps) == 0:
        return []

    sig = np.asarray(sig, dtype=np.float32)
    p10 = float(np.nanpercentile(sig, 10)) if np.isfinite(sig).any() else 0.0
    p90 = float(np.nanpercentile(sig, 90)) if np.isfinite(sig).any() else 0.0
    mid = 0.5 * (p10 + p90)
    refined = []

    for i, (st, bt, en) in enumerate(reps):
        lo = int(bt) + 1
        if i + 1 < len(reps):
            hi = max(lo, int(reps[i + 1][1]) - 1)  # up to just before next top
        else:
            # For the last rep, search to clip end to avoid ending at top.
            hi = T - 1

        if hi > lo and np.isfinite(sig[lo:hi + 1]).any():
            seg = sig[lo:hi + 1]
            new_end = int((np.argmax(seg) + lo) if top_is_low else (np.argmin(seg) + lo))
            # Safety: end must lie in opposite phase from top around signal midline.
            valid_opposite_phase = True
            if np.isfinite(sig[int(bt)]) and np.isfinite(sig[new_end]):
                same_side = ((float(sig[int(bt)]) - mid) * (float(sig[new_end]) - mid)) > 0.0
                if same_side:
                    idx = np.where(np.isfinite(seg))[0]
                    if idx.size > 0:
                        cand = idx + lo
                        opp = [int(c) for c in cand if ((float(sig[int(bt)]) - mid) * (float(sig[int(c)]) - mid)) <= 0.0]
                        if opp:
                            # choose strongest down candidate among opposite-side points
                            if top_is_low:
                                new_end = int(max(opp, key=lambda c: float(sig[c])))
                            else:
                                new_end = int(min(opp, key=lambda c: float(sig[c])))
                        else:
                            valid_opposite_phase = False
                    else:
                        valid_opposite_phase = False
            if not valid_opposite_phase:
                continue
        else:
            new_end = int(en)

        new_start = int(refined[-1][2]) if refined else int(st)
        if new_start < new_end and new_start <= int(bt) <= new_end:
            refined.append((new_start, int(bt), new_end))

    return [(int(st), int(bt), int(en)) for (st, bt, en) in refined if st < en and st <= bt <= en]


def split_merged_lateral_reps(sig, reps, top_is_low, stride, p10, p90, rng):
    """
    Split very long lateral-raise reps when they likely contain 2 reps merged together.
    """
    if len(reps) <= 1:
        return reps

    lengths = [int(en - st) for (st, _, en) in reps]
    med_len = float(np.median(lengths)) if lengths else 0.0
    long_thr = max(int(40 / max(1, stride)), int(1.45 * max(1.0, med_len)))
    long_thr = min(long_thr, int(64 / max(1, stride)))
    min_piece = max(6, int(10 / max(1, stride)))
    min_dist = max(5, int(10 / max(1, stride)))
    prom = max(0.006, 0.06 * max(rng, 1e-6))
    top_gate = (p10 + 0.45 * rng) if top_is_low else (p90 - 0.45 * rng)

    out = []
    for (st, bt, en) in reps:
        rep_len = int(en - st)
        if rep_len < long_thr:
            out.append((int(st), int(bt), int(en)))
            continue

        seg = np.asarray(sig[st:en + 1], dtype=np.float32)
        if not np.isfinite(seg).any():
            out.append((int(st), int(bt), int(en)))
            continue

        if top_is_low:
            tops_raw = find_local_minima(seg, min_dist=min_dist, prominence=prom)
            tops = [int(st + t) for t in tops_raw if np.isfinite(sig[int(st + t)]) and (sig[int(st + t)] <= top_gate)]
            if len(tops) < 2:
                tops = [int(st + t) for t in tops_raw if np.isfinite(sig[int(st + t)]) and (sig[int(st + t)] <= (p10 + 0.65 * rng))]
        else:
            tops_raw = find_local_minima(-seg, min_dist=min_dist, prominence=prom)
            tops = [int(st + t) for t in tops_raw if np.isfinite(sig[int(st + t)]) and (sig[int(st + t)] >= top_gate)]
            if len(tops) < 2:
                tops = [int(st + t) for t in tops_raw if np.isfinite(sig[int(st + t)]) and (sig[int(st + t)] >= (p90 - 0.65 * rng))]

        tops = sorted(tops)
        if len(tops) < 2:
            out.append((int(st), int(bt), int(en)))
            continue

        # Build multiple cycles using consecutive tops.
        cur_st = int(st)
        made = 0
        for j in range(len(tops) - 1):
            t1 = int(tops[j])
            t2 = int(tops[j + 1])
            if t2 - t1 < min_piece:
                continue
            a, b = t1, t2
            if top_is_low:
                split_en = int(np.argmax(sig[a:b + 1]) + a)
            else:
                split_en = int(np.argmin(sig[a:b + 1]) + a)

            if (split_en - cur_st) < min_piece:
                continue
            if cur_st <= t1 <= split_en:
                out.append((int(cur_st), int(t1), int(split_en)))
                cur_st = int(split_en)
                made += 1

        last_top = int(tops[-1])
        if made > 0 and (en - cur_st) >= min_piece and cur_st <= last_top <= en:
            out.append((int(cur_st), int(last_top), int(en)))
        elif made == 0:
            out.append((int(st), int(bt), int(en)))

    cleaned = []
    for (st, bt, en) in out:
        st_i = int(st)
        bt_i = int(bt)
        en_i = int(en)
        if cleaned:
            st_i = int(cleaned[-1][2])
        if st_i < en_i and st_i <= bt_i <= en_i:
            cleaned.append((st_i, bt_i, en_i))
    return cleaned


def reps_by_hysteresis_bench(sig, T, bottom_thr, top_thr, min_gap=10, hold=2):
    """
    Bench press cycles: bottom (low elbow angle) -> top (lockout-ish).
    Count only full bottom->top transitions to avoid splitting one slow
    heavy rep into multiple reps near the bottom.
    """
    reps = []
    state = "seek_bottom"
    last_switch = -10**9
    bottom_idx = None
    below_cnt = 0
    above_cnt = 0

    for t in range(T):
        v = sig[t]
        if np.isnan(v):
            below_cnt = 0
            above_cnt = 0
            continue

        if state == "seek_bottom":
            below_cnt = below_cnt + 1 if v <= bottom_thr else 0
            if below_cnt >= hold and (t - last_switch) >= min_gap:
                bottom_idx = t
                last_switch = t
                state = "seek_top"
                below_cnt = 0
                above_cnt = 0
        else:
            above_cnt = above_cnt + 1 if v >= top_thr else 0
            if above_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                start_idx = reps[-1][2] if reps else 0
                reps.append((start_idx, bottom_idx, top_idx))
                last_switch = t
                state = "seek_bottom"
                below_cnt = 0
                above_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def torso_to_floor_angle_signal(seq):
    """
    Torso angle against floor in degrees:
      - ~90 deg = torso upright
      - ~0 deg  = torso almost parallel to floor
    """
    T = int(seq.shape[0])
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        sx = np.nanmean([lm[11, 0], lm[12, 0]])
        sy = np.nanmean([lm[11, 1], lm[12, 1]])
        hx = np.nanmean([lm[23, 0], lm[24, 0]])
        hy = np.nanmean([lm[23, 1], lm[24, 1]])
        if not np.isfinite([sx, sy, hx, hy]).all():
            continue
        dx = abs(float(sx - hx))
        dy = abs(float(sy - hy))
        sig[t] = float(np.degrees(np.arctan2(dy, dx + 1e-9)))
    return sig


def knee_angle_mean_signal(seq):
    T = int(seq.shape[0])
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]

        vals = []
        for h, k, a in ((23, 25, 27), (24, 26, 28)):
            pts = lm[[h, k, a], :2]
            if np.isnan(pts).any():
                continue
            vals.append(angle_3pts(pts[0], pts[1], pts[2]))
        if vals:
            sig[t] = float(np.mean(vals))
    return sig


def hip_hinge_angle_signal(seq):
    """
    Hip hinge angle in degrees (mean of both sides):
      angle(shoulder - hip - knee)
    Higher ~ more upright, lower ~ deeper hinge.
    """
    T = int(seq.shape[0])
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        vals = []
        for sh, hp, kn in ((11, 23, 25), (12, 24, 26)):
            pts = lm[[sh, hp, kn], :2]
            if np.isnan(pts).any():
                continue
            vals.append(angle_3pts(pts[0], pts[1], pts[2]))
        if vals:
            sig[t] = float(np.mean(vals))
    return sig


def reps_by_hysteresis_rdl(sig, T, top_thr, bottom_thr, min_gap=10, hold=2):
    """
    Romanian deadlift cycles:
      top (upright torso) -> bottom (hinge, torso lowered) -> top.
    """
    reps = []
    state = "seek_top_ready"
    last_switch = -10**9

    top_ready_idx = None
    bottom_idx = None

    above_cnt = 0
    below_cnt = 0

    for t in range(T):
        v = sig[t]
        if np.isnan(v):
            above_cnt = 0
            below_cnt = 0
            continue

        if state == "seek_top_ready":
            above_cnt = above_cnt + 1 if v >= top_thr else 0
            if above_cnt >= hold:
                top_ready_idx = t
                state = "seek_bottom"
                above_cnt = 0
                below_cnt = 0
        elif state == "seek_bottom":
            below_cnt = below_cnt + 1 if v <= bottom_thr else 0
            if below_cnt >= hold and (t - last_switch) >= min_gap:
                bottom_idx = t
                last_switch = t
                state = "seek_top"
                above_cnt = 0
                below_cnt = 0
        else:
            above_cnt = above_cnt + 1 if v >= top_thr else 0
            if above_cnt >= hold and (t - last_switch) >= min_gap:
                top_idx = t
                start_idx = reps[-1][2] if reps else int(max(0, (top_ready_idx if top_ready_idx is not None else 0) - hold + 1))
                reps.append((start_idx, bottom_idx, top_idx))
                last_switch = t
                top_ready_idx = t
                state = "seek_bottom"
                above_cnt = 0
                below_cnt = 0

    return [(int(st), int(bt), int(en)) for (st, bt, en) in reps if st < en and st <= bt <= en]


def merge_adjacent_bench_reps(reps, sig, stride=1):
    """
    Merge false double-splits in bench press when motion is slow near the bottom.
    If two adjacent bottoms are close and there was no clear lockout between them,
    treat them as one rep.
    """
    if len(reps) < 2:
        return reps

    sig = np.asarray(sig, dtype=np.float32)
    p10 = float(np.nanpercentile(sig, 10))
    p90 = float(np.nanpercentile(sig, 90))
    rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0

    # Bench elbow-angle signal is in degrees.
    # "Rise" is the local top elevation between two candidate bottoms.
    min_rise = max(4.0, 0.28 * rng)
    very_low_rise = max(2.5, 0.16 * rng)
    short_gap = max(8, int(18 / max(1, stride)))
    very_short_gap = max(5, int(10 / max(1, stride)))
    heavy_split_gap = max(16, int(44 / max(1, stride)))
    # Boundary between two reps should be close to lockout.
    lockout_thr = max(145.0, p10 + 0.72 * rng)

    merged = [tuple(reps[0])]
    for nxt in reps[1:]:
        st1, bt1, en1 = merged[-1]
        st2, bt2, en2 = nxt

        a = min(int(bt1), int(bt2))
        b = max(int(bt1), int(bt2))
        if b <= a:
            merged[-1] = (st1, bt1, en2)
            continue

        local = sig[a:b + 1]
        if local.size == 0 or not np.isfinite(local).any():
            merged.append(tuple(nxt))
            continue

        local_top = float(np.nanmax(local))
        bottom_level = float(np.nanmin([sig[bt1], sig[bt2]]))
        rise = local_top - bottom_level
        bottom_gap = int(bt2) - int(bt1)
        boundary_top = float(sig[en1]) if np.isfinite(sig[en1]) else local_top
        no_lockout_between = boundary_top < lockout_thr

        # Merge only for clear false split near one prolonged heavy rep:
        # very close bottoms + weak rise and no lockout around boundary.
        should_merge = (
            (no_lockout_between and bottom_gap <= heavy_split_gap and rise < min_rise) or
            (no_lockout_between and bottom_gap <= short_gap and rise < (very_low_rise * 1.4)) or
            (bottom_gap <= very_short_gap and rise < very_low_rise)
        )

        if should_merge:
            new_bottom = int(bt1) if sig[bt1] <= sig[bt2] else int(bt2)
            merged[-1] = (int(st1), new_bottom, int(en2))
        else:
            merged.append(tuple(nxt))

    return [(int(st), int(bt), int(en)) for (st, bt, en) in merged if st < en and st <= bt <= en]


# ----------------- scoring -----------------

def score_squat_rep(seq, start, bottom, end):
    lm = seq[bottom]
    hip_y = np.nanmean([lm[23, 1], lm[24, 1]])
    knee_y = np.nanmean([lm[25, 1], lm[26, 1]])
    insufficient_depth = int(hip_y < knee_y + 0.02)
    errors = {"insufficient_depth": insufficient_depth}
    rep_ok = 1 if sum(errors.values()) == 0 else 0
    return rep_ok, errors


def score_bench_rep(seq, start, bottom, end):
    lm_top = seq[end]
    lw, rw = 15, 16
    if np.isnan(lm_top[[lw, rw], :2]).any():
        uneven_lockout = 0
    else:
        diff = abs(float(lm_top[lw, 1] - lm_top[rw, 1]))
        uneven_lockout = int(diff > 0.04)
    errors = {"uneven_lockout": uneven_lockout}
    rep_ok = 1 if sum(errors.values()) == 0 else 0
    return rep_ok, errors


def score_lat_pulldown_rep(seq, start, bottom, end):
    """
    Минимальный скоринг: в bottom локти ниже плеч.
    """
    lm = seq[bottom]
    sh_y = np.nanmean([lm[11, 1], lm[12, 1]])
    el_y = np.nanmean([lm[13, 1], lm[14, 1]])

    elbows_not_below_shoulders = int(el_y <= sh_y + 0.02)
    errors = {"elbows_not_below_shoulders": elbows_not_below_shoulders}
    rep_ok = 1 if sum(errors.values()) == 0 else 0
    return rep_ok, errors


def score_lateral_raises_rep(seq, start, bottom, end):
    vis_thr = 0.35
    lm_top = seq[bottom]
    W = seq[start:end + 1]
    sh_vals = [float(lm_top[i, 1]) for i in (11, 12) if np.isfinite(lm_top[i, 1])]
    sh_y = float(np.mean(sh_vals)) if sh_vals else np.nan

    side_y = []
    for wr, el in ((15, 13), (16, 14)):
        y_wr, v_wr = lm_top[wr, 1], lm_top[wr, 3]
        y_el, v_el = lm_top[el, 1], lm_top[el, 3]
        if np.isfinite(y_wr) and np.isfinite(v_wr) and v_wr >= vis_thr:
            side_y.append(float(y_wr))
        elif np.isfinite(y_el) and np.isfinite(v_el) and v_el >= vis_thr:
            side_y.append(float(y_el))

    if len(side_y) == 2 and np.isfinite(sh_y):
        mean_top_y = 0.5 * (side_y[0] + side_y[1])
        insufficient_rom = int(mean_top_y > sh_y + 0.04)
        asymmetry = int(abs(float(side_y[0] - side_y[1])) > 0.05)
    elif len(side_y) == 1 and np.isfinite(sh_y):
        insufficient_rom = int(side_y[0] > sh_y + 0.04)
        asymmetry = 0
    else:
        insufficient_rom = 0
        asymmetry = 0

    eaL = elbow_angle_signal(seq[start:end + 1], "left")
    eaR = elbow_angle_signal(seq[start:end + 1], "right")
    mins = []
    if np.isfinite(eaL).any():
        mins.append(float(np.nanmin(eaL)))
    if np.isfinite(eaR).any():
        mins.append(float(np.nanmin(eaR)))
    too_much_elbow_bend = int(bool(mins) and (min(mins) < 120.0))

    depth = lateral_raise_depth_with_fallback(W, vis_thr=vis_thr)
    if np.isfinite(depth).any():
        top_d = float(np.nanmin(depth))
        end_d = float(depth[-1]) if np.isfinite(depth[-1]) else float(np.nanmedian(depth))
        incomplete_lowering = int((end_d - top_d) < 0.05)
    else:
        incomplete_lowering = 0

    sh_y_series = np.full((W.shape[0],), np.nan, dtype=np.float32)
    for t in range(W.shape[0]):
        vals = []
        for i in (11, 12):
            y = W[t, i, 1]
            if np.isfinite(y):
                vals.append(float(y))
        if vals:
            sh_y_series[t] = float(np.mean(vals))
    if np.isfinite(sh_y_series).any():
        sh_top = float(sh_y_series[max(0, min(len(sh_y_series) - 1, bottom - start))])
        sh_med = float(np.nanmedian(sh_y_series))
        shoulder_shrug = int((sh_med - sh_top) > 0.018)
    else:
        shoulder_shrug = 0

    hip_x = np.full((W.shape[0],), np.nan, dtype=np.float32)
    for t in range(W.shape[0]):
        vals = []
        for i in (23, 24):
            x = W[t, i, 0]
            if np.isfinite(x):
                vals.append(float(x))
        if vals:
            hip_x[t] = float(np.mean(vals))
    torso_sway = int(np.isfinite(hip_x).any() and (float(np.nanstd(hip_x)) > 0.02))

    errors = {
        "insufficient_rom": insufficient_rom,
        "asymmetry": asymmetry,
        "too_much_elbow_bend": too_much_elbow_bend,
        "incomplete_lowering": incomplete_lowering,
        "shoulder_shrug": shoulder_shrug,
        "torso_sway": torso_sway,
    }
    rep_ok = 1 if sum(errors.values()) == 0 else 0
    return rep_ok, errors


def score_deadlift_rep(seq, start, bottom, end):
    W = seq[start:end + 1]
    if W.shape[0] == 0:
        return 1, {}

    torso_sig = torso_to_floor_angle_signal(W)
    knee_sig = knee_angle_mean_signal(W)

    start_torso = float(torso_sig[0]) if np.isfinite(torso_sig[0]) else np.nan
    end_torso = float(torso_sig[-1]) if np.isfinite(torso_sig[-1]) else np.nan
    bot_torso = float(torso_sig[max(0, min(W.shape[0] - 1, bottom - start))]) if np.isfinite(torso_sig).any() else np.nan

    insufficient_hip_hinge = int(np.isfinite(bot_torso) and (bot_torso > 50.0))
    incomplete_lockout = int(np.isfinite(end_torso) and (end_torso < 70.0))
    excessive_knee_bend = int(np.isfinite(knee_sig).any() and (float(np.nanmin(knee_sig)) < 145.0))
    rounded_back = int(np.isfinite([start_torso, end_torso]).all() and (abs(end_torso - start_torso) > 18.0))

    lm_bot = seq[bottom]
    wh_dist = np.nanmean([
        abs(float(lm_bot[15, 0] - lm_bot[23, 0])) if np.isfinite(lm_bot[[15, 23], 0]).all() else np.nan,
        abs(float(lm_bot[16, 0] - lm_bot[24, 0])) if np.isfinite(lm_bot[[16, 24], 0]).all() else np.nan,
    ])
    bar_too_far_from_body = int(np.isfinite(wh_dist) and (wh_dist > 0.22))

    errors = {
        "insufficient_hip_hinge": insufficient_hip_hinge,
        "rounded_back": rounded_back,
        "excessive_knee_bend": excessive_knee_bend,
        "incomplete_lockout": incomplete_lockout,
        "bar_too_far_from_body": bar_too_far_from_body,
    }
    rep_ok = 1 if sum(errors.values()) == 0 else 0
    return rep_ok, errors


# ----------------- summary -----------------

def summarize_set(rep_results):
    n = len(rep_results)
    if n == 0:
        return {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

    good = sum(int(r.get("rep_ok", 0)) for r in rep_results)
    err_counts = {}
    for r in rep_results:
        for k, v in r.get("errors", {}).items():
            err_counts[k] = err_counts.get(k, 0) + int(v)

    top_errs = sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "reps": int(n),
        "good_reps": int(good),
        "good_rate": float(good) / float(n),
        "errors": top_errs
    }


# ----------------- main API -----------------

def analyze_reps(seq, exercise_name, stride=2):
    """
    seq: (T,33,4) normalized landmarks
    returns: (rep_results, summary)
    """
    T = int(seq.shape[0])
    if T <= 0:
        return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

    ex = str(exercise_name).lower()
    if ex == "lat_pulldown":
        ex = "latpulldown"
    if ex in ("lateral_raise", "lateralraises"):
        ex = "lateral_raises"
    if ex in ("deadlift", "dead_lift", "romanian_deadlift", "romanian deadlift", "rdl"):
        ex = "romanian_deadlift"

    # ---- SPECIAL: LAT PULLDOWN (лучше не через get_primary_signal)
    if ex == "latpulldown":
        sigA = smooth_1d(elbow_angle_signal_both(seq), win=11)
        gate = elbows_below_shoulders_signal(seq, margin=0.02)

        p10 = np.nanpercentile(sigA, 10)
        p90 = np.nanpercentile(sigA, 90)
        rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
        if rng < 6.0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        bottom_thr = p10 + 0.30 * rng
        top_thr = p10 + 0.70 * rng
        min_gap = max(6, int(10 / max(1, stride)))

        reps = reps_by_hysteresis_lat(
            sigA, gate, T,
            bottom_thr=bottom_thr,
            top_thr=top_thr,
            min_gap=min_gap,
            hold=2,
            gate_hold=2
        )

        min_len = max(10, int(18 / max(1, stride)))
        reps = [(st, bt, en) for (st, bt, en) in reps if (en - st) >= min_len]

        if len(reps) == 0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        results = []
        for i, (st, bt, en) in enumerate(reps, start=1):
            rep_ok, errors = score_lat_pulldown_rep(seq, st, bt, en)
            results.append({
                "rep": int(i),
                "start": int(st),
                "bottom": int(bt),
                "end": int(en),
                "rep_ok": int(rep_ok),
                "errors": errors
            })
        return results, summarize_set(results)

    # ---- SPECIAL: BENCH PRESS
    if ex == "bench_press":
        sig = smooth_1d(elbow_angle_signal_both(seq), win=11)
        if not np.isfinite(sig).any():
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        p10 = float(np.nanpercentile(sig, 10))
        p90 = float(np.nanpercentile(sig, 90))
        rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
        if rng < 6.0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        bottom_thr = p10 + 0.30 * rng
        top_thr = p10 + 0.62 * rng
        min_gap = max(5, int(8 / max(1, stride)))

        reps = reps_by_hysteresis_bench(
            sig, T,
            bottom_thr=bottom_thr,
            top_thr=top_thr,
            min_gap=min_gap,
            hold=1,
        )

        min_len = max(8, int(12 / max(1, stride)))
        reps_new = [(st, bt, en) for (st, bt, en) in reps if (en - st) >= min_len]

        # Old minima segmentation (always compute for auto-choice policy).
        min_dist = max(10, int(36 / max(1, stride)))
        min_sep = max(6, int(0.6 * min_dist))
        bottoms = find_local_minima(sig, min_dist=min_dist, prominence=8.0)
        bottoms = dedup_bottoms(bottoms, sig, min_sep=min_sep)
        tops = find_tops_between_bottoms(sig, bottoms)
        reps_old = reps_from_bottoms_and_tops(bottoms, tops, T)
        reps_old = merge_adjacent_bench_reps(reps_old, sig, stride=stride)
        reps_old = [(st, bt, en) for (st, bt, en) in reps_old if (en - st) >= max(10, int(20 / max(1, stride)))]

        # Selection policy requested by user:
        # 1) if new == 0 -> use old
        # 2) if new > 0 and new < old -> use new
        # 3) else -> use old
        if len(reps_new) == 0:
            reps = reps_old
        elif len(reps_new) < len(reps_old):
            reps = reps_new
        else:
            reps = reps_old

        if len(reps) == 0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        results = []
        for i, (st, bt, en) in enumerate(reps, start=1):
            rep_ok, errors = score_bench_rep(seq, st, bt, en)
            results.append({
                "rep": int(i),
                "start": int(st),
                "bottom": int(bt),
                "end": int(en),
                "rep_ok": int(rep_ok),
                "errors": errors
            })
        return results, summarize_set(results)

    # ---- SPECIAL: LATERAL RAISES (count only full raise to shoulder line)
    if ex == "lateral_raises":
        up = smooth_1d(build_lateral_upscore(seq), win=5)
        if not np.isfinite(up).any():
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        p10 = float(np.nanpercentile(up, 10))
        p90 = float(np.nanpercentile(up, 90))
        rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
        if rng < 0.10:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        top_thr = min(0.83, p10 + 0.57 * rng)
        down_thr = max(0.10, p10 + 0.22 * rng)
        min_len = max(5, int(7 / max(1, stride)))
        min_amp = max(0.05, 0.12 * rng)
        end_gate_soft = max(down_thr + max(0.06, 0.12 * rng), p10 + 0.48 * rng)
        end_gate_hard = max(down_thr + max(0.10, 0.16 * rng), p10 + 0.52 * rng)

        downs = _down_anchors_from_upscore(up, down_thr=down_thr, stride=stride)
        reps = []
        for i in range(len(downs) - 1):
            st = int(downs[i])
            en = int(downs[i + 1])
            if (en - st) < min_len:
                continue
            lo = st + max(1, (en - st) // 8)
            hi = en - max(1, (en - st) // 8)
            if hi <= lo or not np.isfinite(up[lo:hi + 1]).any():
                continue
            bt = int(np.argmax(up[lo:hi + 1]) + lo)  # top point
            if not np.isfinite(up[bt]):
                continue
            if up[bt] < top_thr:
                continue
            base = 0.5 * (float(up[st]) + float(up[en]))
            if (float(up[bt]) - base) < min_amp:
                continue
            # hard rule: rep must end in down phase, never at top
            if up[en] > end_gate_soft:
                continue
            reps.append((st, bt, en))

        reps = _split_long_lateral_cycles(up, reps, top_thr=top_thr, down_thr=down_thr, stride=stride)

        # final sanitation for down->top->down
        cleaned = []
        for (st, bt, en) in reps:
            st = int(st)
            bt = int(bt)
            en = int(en)
            if not (st < bt < en):
                continue
            if not (np.isfinite(up[st]) and np.isfinite(up[bt]) and np.isfinite(up[en])):
                continue
            if up[bt] < top_thr:
                continue
            if max(float(up[st]), float(up[en])) > end_gate_hard:
                continue
            cleaned.append((st, bt, en))
        reps = cleaned

        chained = []
        for (st, bt, en) in reps:
            st_i = int(chained[-1][2]) if chained else int(st)
            if st_i < int(en) and st_i <= int(bt) <= int(en):
                chained.append((st_i, int(bt), int(en)))
        reps = chained
        if len(reps) == 0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        results = []
        for i, (st, bt, en) in enumerate(reps, start=1):
            rep_ok, errors = score_lateral_raises_rep(seq, st, bt, en)
            results.append({
                "rep": int(i),
                "start": int(st),
                "bottom": int(bt),
                "end": int(en),
                "rep_ok": int(rep_ok),
                "errors": errors
            })
        return results, summarize_set(results)

    # ---- SPECIAL: ROMANIAN DEADLIFT (top -> hinge near parallel -> top)
    if ex == "romanian_deadlift":
        sig_hip = smooth_1d(hip_hinge_angle_signal(seq), win=7)
        sig_torso = smooth_1d(torso_to_floor_angle_signal(seq), win=7)

        hip_ok = np.isfinite(sig_hip).any()
        torso_ok = np.isfinite(sig_torso).any()
        if not hip_ok and not torso_ok:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        hip_rng = float(np.nanpercentile(sig_hip, 90) - np.nanpercentile(sig_hip, 10)) if hip_ok else 0.0
        torso_rng = float(np.nanpercentile(sig_torso, 90) - np.nanpercentile(sig_torso, 10)) if torso_ok else 0.0

        # Prefer hip-hinge angle when available; it's more robust to camera tilt.
        sig = sig_hip if (hip_ok and (hip_rng >= max(8.0, torso_rng * 0.75) or not torso_ok)) else sig_torso

        p10 = float(np.nanpercentile(sig, 10))
        p90 = float(np.nanpercentile(sig, 90))
        rng = float(p90 - p10) if np.isfinite(p90 - p10) else 0.0
        if rng < 6.0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        bottom_thr = p10 + 0.30 * rng
        top_thr = p10 + 0.68 * rng
        min_gap = max(6, int(10 / max(1, stride)))
        min_len = max(8, int(12 / max(1, stride)))
        min_amp = max(5.0, 0.20 * rng)

        reps = reps_by_hysteresis_rdl(
            sig, T,
            top_thr=top_thr,
            bottom_thr=bottom_thr,
            min_gap=min_gap,
            hold=2,
        )

        # Fallback: extrema-based split for clips where hysteresis misses cycles.
        if len(reps) == 0:
            min_dist = max(8, int(20 / max(1, stride)))
            min_sep = max(5, int(0.6 * min_dist))
            prom = max(3.0, 0.18 * rng)
            bottoms = find_local_minima(sig, min_dist=min_dist, prominence=prom)
            bottoms = dedup_bottoms(bottoms, sig, min_sep=min_sep)
            tops = find_tops_between_bottoms(sig, bottoms)
            reps = reps_from_bottoms_and_tops(bottoms, tops, T)

        filtered = []
        for (st, bt, en) in reps:
            if (en - st) < min_len:
                continue
            if not (np.isfinite(sig[st]) and np.isfinite(sig[bt]) and np.isfinite(sig[en])):
                continue
            amp = min(float(sig[st]), float(sig[en])) - float(sig[bt])
            if amp < min_amp:
                continue
            if float(sig[bt]) > (bottom_thr + 0.06 * rng):
                continue
            filtered.append((int(st), int(bt), int(en)))
        reps = filtered

        if len(reps) == 0:
            return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

        results = []
        for i, (st, bt, en) in enumerate(reps, start=1):
            rep_ok, errors = score_deadlift_rep(seq, st, bt, en)
            results.append({
                "rep": int(i),
                "start": int(st),
                "bottom": int(bt),
                "end": int(en),
                "rep_ok": int(rep_ok),
                "errors": errors
            })
        return results, summarize_set(results)

    # ---- GENERIC: take primary signal from extract_signals
    sig, meta = get_primary_signal(seq, ex)
    if sig is None:
        return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

    sig = smooth_1d(sig, win=int(meta.get("smooth_win", 11)))

    min_dist = max(10, int(36 / max(1, stride)))
    min_sep = max(6, int(0.6 * min_dist))

    bottom_is_min = bool(meta.get("bottom_is_min", True))
    prom = float(meta.get("prominence", 8.0))

    if bottom_is_min:
        bottoms = find_local_minima(sig, min_dist=min_dist, prominence=prom)
        bottoms = dedup_bottoms(bottoms, sig, min_sep=min_sep)
        tops = find_tops_between_bottoms(sig, bottoms)
    else:
        bottoms = find_local_minima(-sig, min_dist=min_dist, prominence=prom)
        bottoms = dedup_bottoms(bottoms, -sig, min_sep=min_sep)
        tops = find_tops_between_bottoms(-sig, bottoms)

    reps = reps_from_bottoms_and_tops(bottoms, tops, T)
    min_len = max(10, int(20 / max(1, stride)))
    reps = [(st, bt, en) for (st, bt, en) in reps if (en - st) >= min_len]

    if len(reps) == 0:
        return [], {"reps": 0, "good_reps": 0, "good_rate": 0.0, "errors": []}

    results = []
    for i, (st, bt, en) in enumerate(reps, start=1):
        if ex == "squat":
            rep_ok, errors = score_squat_rep(seq, st, bt, en)
        elif ex == "bench_press":
            rep_ok, errors = score_bench_rep(seq, st, bt, en)
        elif ex == "lateral_raises":
            rep_ok, errors = score_lateral_raises_rep(seq, st, bt, en)
        elif ex in ("romanian_deadlift", "deadlift"):
            rep_ok, errors = score_deadlift_rep(seq, st, bt, en)
        else:
            # неизвестное упражнение: не падаем
            rep_ok, errors = 1, {}

        results.append({
            "rep": int(i),
            "start": int(st),
            "bottom": int(bt),
            "end": int(en),
            "rep_ok": int(rep_ok),
            "errors": errors
        })

    return results, summarize_set(results)
