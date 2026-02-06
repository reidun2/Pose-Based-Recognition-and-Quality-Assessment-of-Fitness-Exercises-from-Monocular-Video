# reps.py
import numpy as np
from extract_signals import get_primary_signal  # единый источник "primary signal" для squat/bench/и т.д.


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
