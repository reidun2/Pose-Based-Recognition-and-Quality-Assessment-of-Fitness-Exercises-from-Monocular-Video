import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CV_REPORT = BASE_DIR / "data" / "processed" / "cv_report.json"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed" / "eval_eda"


def load_json(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def stats(values):
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return {}
    arr = np.array(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "cv": float(std / mean) if mean != 0 else None,
    }


def sort_labels(labels):
    items = [str(x) for x in labels]
    try:
        return [str(x) for x in sorted(items, key=lambda s: int(float(s)))]
    except Exception:
        return sorted(items)


def load_exercise_map_from_dataset(dataset_path):
    p = Path(dataset_path)
    if not p.is_file():
        return {}
    try:
        data = np.load(p, allow_pickle=True)
    except Exception:
        return {}
    if "ex_map" not in data.files:
        return {}
    raw = data["ex_map"].item()
    if not isinstance(raw, dict):
        return {}
    inv = {}
    for name, idx in raw.items():
        try:
            inv[str(int(idx))] = str(name)
        except Exception:
            continue
    return inv


def parse_class_report_rows(folds, report_key, class_name_fn):
    rows = []
    for fold in folds:
        fold_id = int(fold.get("fold", 0))
        rep = fold.get(report_key, {})
        if not isinstance(rep, dict):
            continue
        for label, m in rep.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            if not isinstance(m, dict):
                continue
            rows.append(
                {
                    "fold": fold_id,
                    "class_key": str(label),
                    "class_name": class_name_fn(label),
                    "precision": to_float(m.get("precision")),
                    "recall": to_float(m.get("recall")),
                    "f1_score": to_float(m.get("f1-score")),
                    "support": to_float(m.get("support")),
                }
            )
    return rows


def summarize_class_rows(rows):
    bucket = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cn = r["class_name"]
        bucket[cn]["precision"].append(r["precision"])
        bucket[cn]["recall"].append(r["recall"])
        bucket[cn]["f1_score"].append(r["f1_score"])
        bucket[cn]["support"].append(r["support"])

    out = []
    for cn in sorted(bucket.keys()):
        out.append(
            {
                "class_name": cn,
                "precision_mean": stats(bucket[cn]["precision"]).get("mean"),
                "precision_std": stats(bucket[cn]["precision"]).get("std"),
                "recall_mean": stats(bucket[cn]["recall"]).get("mean"),
                "recall_std": stats(bucket[cn]["recall"]).get("std"),
                "f1_mean": stats(bucket[cn]["f1_score"]).get("mean"),
                "f1_std": stats(bucket[cn]["f1_score"]).get("std"),
                "support_mean": stats(bucket[cn]["support"]).get("mean"),
                "support_std": stats(bucket[cn]["support"]).get("std"),
            }
        )
    return out


def report_f1(report):
    if not isinstance(report, dict):
        return {}
    out = {}
    macro = report.get("macro avg")
    weighted = report.get("weighted avg")
    if isinstance(macro, dict):
        out["f1_macro"] = to_float(macro.get("f1-score"))
    if isinstance(weighted, dict):
        out["f1_weighted"] = to_float(weighted.get("f1-score"))
    return out


def write_csv(path, rows, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_matrix_csv(path, matrix, labels):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix)
    lbl = [str(x) for x in labels]
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred"] + lbl)
        for i, row_name in enumerate(lbl):
            row_vals = []
            for v in arr[i]:
                fv = float(v)
                if abs(fv - round(fv)) < 1e-12:
                    row_vals.append(int(round(fv)))
                else:
                    row_vals.append(f"{fv:.6f}")
            w.writerow([row_name] + row_vals)


def normalize_confusion(cm_counts):
    if cm_counts is None:
        return None
    arr = np.asarray(cm_counts, dtype=np.float64)
    row_sum = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    np.divide(arr, row_sum, out=out, where=row_sum > 0)
    return out


def aggregate_confusion_from_folds(folds, matrix_key, labels_key, master_labels):
    master = [str(x) for x in master_labels]
    idx = {lab: i for i, lab in enumerate(master)}
    agg = np.zeros((len(master), len(master)), dtype=np.float64)
    used_folds = 0

    for f in folds:
        mat = f.get(matrix_key)
        if not isinstance(mat, list):
            continue
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            continue

        fold_labels_raw = f.get(labels_key, [])
        if isinstance(fold_labels_raw, list) and len(fold_labels_raw) == arr.shape[0]:
            fold_labels = [str(x) for x in fold_labels_raw]
        elif arr.shape[0] == len(master):
            fold_labels = master
        else:
            continue

        used_folds += 1
        for i_true, l_true in enumerate(fold_labels):
            ti = idx.get(str(l_true))
            if ti is None:
                continue
            for j_pred, l_pred in enumerate(fold_labels):
                pj = idx.get(str(l_pred))
                if pj is None:
                    continue
                agg[ti, pj] += arr[i_true, j_pred]

    if used_folds == 0:
        return None, 0
    return agg.astype(np.int64), used_folds


def make_heatmap(ax, matrix, labels, title, cmap="Blues", normalized=False):
    arr = np.asarray(matrix, dtype=np.float64)
    vmax = 1.0 if normalized else None
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    if arr.size <= 400:
        threshold = (arr.max() * 0.6) if arr.size else 0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                txt = f"{v:.2f}" if normalized else f"{int(round(v))}"
                color = "white" if v > threshold else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    return im


def maybe_make_confusion_plots(
    out_dir,
    ex_cm_counts,
    ex_cm_norm,
    ex_labels,
    form_cm_counts,
    form_cm_norm,
    form_labels,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {
            "plots_created": False,
            "reason": "matplotlib is not available",
            "files": [],
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    if ex_cm_counts is not None and ex_cm_norm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        im1 = make_heatmap(
            axes[0],
            ex_cm_counts,
            ex_labels,
            "Exercise confusion (counts)",
            cmap="YlGnBu",
            normalized=False,
        )
        im2 = make_heatmap(
            axes[1],
            ex_cm_norm,
            ex_labels,
            "Exercise confusion (row-normalized)",
            cmap="YlOrRd",
            normalized=True,
        )
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = out_dir / "cv_confusion_exercise.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(str(p))

    if form_cm_counts is not None and form_cm_norm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
        im1 = make_heatmap(
            axes[0],
            form_cm_counts,
            form_labels,
            "Form confusion (counts)",
            cmap="PuBuGn",
            normalized=False,
        )
        im2 = make_heatmap(
            axes[1],
            form_cm_norm,
            form_labels,
            "Form confusion (row-normalized)",
            cmap="OrRd",
            normalized=True,
        )
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = out_dir / "cv_confusion_form.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(str(p))

    return {
        "plots_created": len(files) > 0,
        "reason": (
            "" if len(files) > 0 else "confusion matrices are not present in cv_report"
        ),
        "files": files,
    }


def maybe_make_plots(out_dir, fold_rows, ex_class_summary, form_class_summary):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {
            "plots_created": False,
            "reason": "matplotlib is not available",
            "files": [],
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    folds = [int(r["fold"]) for r in fold_rows]
    ex_acc = [to_float(r.get("exercise_accuracy")) for r in fold_rows]
    form_acc = [to_float(r.get("form_accuracy")) for r in fold_rows]
    err_micro = [to_float(r.get("error_f1_micro")) for r in fold_rows]
    err_macro = [to_float(r.get("error_f1_macro")) for r in fold_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(folds, ex_acc, marker="o", label="exercise_accuracy")
    ax.plot(folds, form_acc, marker="o", label="form_accuracy")
    ax.plot(folds, err_micro, marker="o", label="error_f1_micro")
    ax.plot(folds, err_macro, marker="o", label="error_f1_macro")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(folds)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "cv_metrics_by_fold.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    metric_box_data = [ex_acc, form_acc, err_micro, err_macro]
    ax.boxplot(
        metric_box_data,
        tick_labels=["ex_acc", "form_acc", "err_f1_micro", "err_f1_macro"],
        showmeans=True,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Metric distribution across folds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "cv_metrics_boxplot.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sc = ax.scatter(
        ex_acc,
        form_acc,
        c=folds,
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )
    for i, fid in enumerate(folds):
        ax.text(ex_acc[i] + 0.002, form_acc[i] + 0.002, f"F{fid}", fontsize=9)
    ax.set_xlim(max(0.0, min(ex_acc) - 0.03), min(1.0, max(ex_acc) + 0.03))
    ax.set_ylim(max(0.0, min(form_acc) - 0.05), min(1.0, max(form_acc) + 0.05))
    ax.set_xlabel("Exercise accuracy")
    ax.set_ylabel("Form accuracy")
    ax.set_title("Fold-wise exercise vs form")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Fold")
    fig.tight_layout()
    p = out_dir / "cv_exercise_vs_form_scatter.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    # Pairwise scatter plots for key fold-level relations.
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    pair_specs = [
        ("best_val_loss", "exercise_accuracy"),
        ("best_val_loss", "form_accuracy"),
        ("epochs_ran", "exercise_accuracy"),
        ("error_f1_micro", "error_f1_macro"),
    ]
    fold_metric = {
        "fold": folds,
        "exercise_accuracy": ex_acc,
        "form_accuracy": form_acc,
        "error_f1_micro": err_micro,
        "error_f1_macro": err_macro,
        "best_val_loss": [to_float(r.get("best_val_loss")) for r in fold_rows],
        "epochs_ran": [to_float(r.get("epochs_ran")) for r in fold_rows],
    }
    for ax, (xk, yk) in zip(axes.ravel(), pair_specs):
        xs = fold_metric[xk]
        ys = fold_metric[yk]
        ax.scatter(
            xs, ys, c=folds, cmap="plasma", s=70, edgecolor="black", linewidth=0.5
        )
        for i, fid in enumerate(folds):
            if np.isfinite(xs[i]) and np.isfinite(ys[i]):
                ax.text(xs[i], ys[i], f" F{fid}", fontsize=8)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "cv_metric_pair_scatter.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    # Correlation heatmap across fold metrics.
    corr_keys = [
        "exercise_accuracy",
        "form_accuracy",
        "error_f1_micro",
        "error_f1_macro",
        "best_val_loss",
        "epochs_ran",
    ]
    mat = np.array(
        [[to_float(r.get(k)) for k in corr_keys] for r in fold_rows], dtype=np.float64
    )
    corr = np.corrcoef(mat, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr_keys)))
    ax.set_yticks(range(len(corr_keys)))
    ax.set_xticklabels(corr_keys, rotation=30, ha="right")
    ax.set_yticklabels(corr_keys)
    ax.set_title("Fold metrics correlation")
    for i in range(len(corr_keys)):
        for j in range(len(corr_keys)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = out_dir / "cv_metric_correlation_heatmap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    # Bubble scatter: class quality vs support.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ex_prec = [to_float(r.get("precision_mean")) for r in ex_class_summary]
    ex_rec = [to_float(r.get("recall_mean")) for r in ex_class_summary]
    ex_sup = [to_float(r.get("support_mean")) for r in ex_class_summary]
    ex_names = [r.get("class_name", "") for r in ex_class_summary]
    ex_sizes = [max(50.0, s * 22.0) for s in ex_sup]
    axes[0].scatter(
        ex_rec,
        ex_prec,
        s=ex_sizes,
        alpha=0.75,
        c="#2c7fb8",
        edgecolor="black",
        linewidth=0.5,
    )
    for i, name in enumerate(ex_names):
        axes[0].text(ex_rec[i] + 0.003, ex_prec[i] + 0.003, name, fontsize=8)
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("recall_mean")
    axes[0].set_ylabel("precision_mean")
    axes[0].set_title("Exercise class scatter (size=support)")
    axes[0].grid(alpha=0.3)

    form_prec = [to_float(r.get("precision_mean")) for r in form_class_summary]
    form_rec = [to_float(r.get("recall_mean")) for r in form_class_summary]
    form_sup = [to_float(r.get("support_mean")) for r in form_class_summary]
    form_names = [r.get("class_name", "") for r in form_class_summary]
    form_sizes = [max(80.0, s * 6.0) for s in form_sup]
    axes[1].scatter(
        form_rec,
        form_prec,
        s=form_sizes,
        alpha=0.75,
        c="#f03b20",
        edgecolor="black",
        linewidth=0.5,
    )
    for i, name in enumerate(form_names):
        axes[1].text(form_rec[i] + 0.003, form_prec[i] + 0.003, name, fontsize=9)
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("recall_mean")
    axes[1].set_ylabel("precision_mean")
    axes[1].set_title("Form class scatter (size=support)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    p = out_dir / "cv_class_quality_scatter.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.35
    x = np.arange(len(ex_class_summary))
    ex_labels = [r["class_name"] for r in ex_class_summary]
    ex_recall = [to_float(r.get("recall_mean")) for r in ex_class_summary]
    ex_f1 = [to_float(r.get("f1_mean")) for r in ex_class_summary]
    ax.bar(x - width / 2, ex_recall, width=width, label="recall_mean")
    ax.bar(x + width / 2, ex_f1, width=width, label="f1_mean")
    ax.set_xticks(x)
    ax.set_xticklabels(ex_labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Exercise per-class CV mean")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "cv_exercise_per_class_mean.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    form_labels = [r["class_name"] for r in form_class_summary]
    form_recall = [to_float(r.get("recall_mean")) for r in form_class_summary]
    form_f1 = [to_float(r.get("f1_mean")) for r in form_class_summary]
    x = np.arange(len(form_class_summary))
    ax.bar(x - width / 2, form_recall, width=width, label="recall_mean")
    ax.bar(x + width / 2, form_f1, width=width, label="f1_mean")
    ax.set_xticks(x)
    ax.set_xticklabels(form_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Form per-class CV mean")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "cv_form_per_class_mean.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.8))
    ex_std = [to_float(r.get("recall_std")) for r in ex_class_summary]
    axes[0].bar(ex_labels, ex_std, color="#d95f02")
    axes[0].set_title("Exercise recall std by class")
    axes[0].set_ylim(0.0, max(0.05, max(ex_std) + 0.02))
    axes[0].grid(axis="y", alpha=0.3)

    form_std = [to_float(r.get("recall_std")) for r in form_class_summary]
    axes[1].bar(form_labels, form_std, color="#1b9e77")
    axes[1].set_title("Form recall std by class")
    axes[1].set_ylim(0.0, max(0.05, max(form_std) + 0.02))
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = out_dir / "cv_recall_stability.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(str(p))

    return {"plots_created": True, "reason": "", "files": files}


def main():
    p = argparse.ArgumentParser(description="EDA for cross-validation report.")
    p.add_argument("--cv_report", default=str(DEFAULT_CV_REPORT))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    cv_path = Path(args.cv_report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = load_json(cv_path)
    folds = report.get("folds", [])
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"No folds in CV report: {cv_path}")

    dataset_path = report.get("data", "")
    ex_map_inv = load_exercise_map_from_dataset(dataset_path)

    def ex_name(label):
        s = str(label)
        return ex_map_inv.get(
            s,
            ex_map_inv.get(
                str(int(float(s))) if s.replace(".", "", 1).isdigit() else s, s
            ),
        )

    def form_name(label):
        s = str(label)
        if s in ("0", "0.0"):
            return "incorrect"
        if s in ("1", "1.0"):
            return "correct"
        return s

    fold_rows = []
    for f in folds:
        ex_f1 = report_f1(f.get("exercise_report", {}))
        form_f1 = report_f1(f.get("form_report", {}))
        fold_rows.append(
            {
                "fold": int(f.get("fold", 0)),
                "exercise_accuracy": to_float(f.get("exercise_accuracy")),
                "exercise_f1_macro": to_float(ex_f1.get("f1_macro")),
                "exercise_f1_weighted": to_float(ex_f1.get("f1_weighted")),
                "form_accuracy": to_float(f.get("form_accuracy")),
                "form_f1_macro": to_float(form_f1.get("f1_macro")),
                "form_f1_weighted": to_float(form_f1.get("f1_weighted")),
                "error_f1_micro": to_float(f.get("error_f1_micro")),
                "error_f1_macro": to_float(f.get("error_f1_macro")),
                "best_val_loss": to_float(f.get("best_val_loss")),
                "epochs_ran": to_float(f.get("epochs_ran")),
                "n_train": to_float(f.get("n_train")),
                "n_val": to_float(f.get("n_val")),
            }
        )

    fold_rows = sorted(fold_rows, key=lambda x: x["fold"])

    ex_rows = parse_class_report_rows(folds, "exercise_report", ex_name)
    form_rows = parse_class_report_rows(folds, "form_report", form_name)
    ex_summary = summarize_class_rows(ex_rows)
    form_summary = summarize_class_rows(form_rows)

    metrics = {
        "exercise_accuracy": [r["exercise_accuracy"] for r in fold_rows],
        "exercise_f1_macro": [r["exercise_f1_macro"] for r in fold_rows],
        "exercise_f1_weighted": [r["exercise_f1_weighted"] for r in fold_rows],
        "form_accuracy": [r["form_accuracy"] for r in fold_rows],
        "form_f1_macro": [r["form_f1_macro"] for r in fold_rows],
        "form_f1_weighted": [r["form_f1_weighted"] for r in fold_rows],
        "error_f1_micro": [r["error_f1_micro"] for r in fold_rows],
        "error_f1_macro": [r["error_f1_macro"] for r in fold_rows],
        "best_val_loss": [r["best_val_loss"] for r in fold_rows],
        "epochs_ran": [r["epochs_ran"] for r in fold_rows],
    }
    metric_stats = {k: stats(v) for k, v in metrics.items()}

    hardest_ex = sorted(
        ex_summary, key=lambda r: (to_float(r["recall_mean"]), to_float(r["f1_mean"]))
    )[:3]
    hardest_form = sorted(
        form_summary, key=lambda r: (to_float(r["recall_mean"]), to_float(r["f1_mean"]))
    )

    best_ex_fold = max(fold_rows, key=lambda r: to_float(r["exercise_accuracy"]))
    worst_ex_fold = min(fold_rows, key=lambda r: to_float(r["exercise_accuracy"]))
    best_form_fold = max(fold_rows, key=lambda r: to_float(r["form_accuracy"]))
    worst_form_fold = min(fold_rows, key=lambda r: to_float(r["form_accuracy"]))

    ex_conf_labels = []
    for f in folds:
        lbl = f.get("exercise_confusion_labels")
        if isinstance(lbl, list) and len(lbl) > 0:
            ex_conf_labels = [str(x) for x in lbl]
            break
    if not ex_conf_labels:
        ex_conf_labels = sort_labels([r["class_key"] for r in ex_rows])

    form_conf_labels = []
    for f in folds:
        lbl = f.get("form_confusion_labels")
        if isinstance(lbl, list) and len(lbl) > 0:
            form_conf_labels = [str(x) for x in lbl]
            break
    if not form_conf_labels:
        form_conf_labels = sort_labels([r["class_key"] for r in form_rows]) or [
            "0",
            "1",
        ]

    ex_cm_counts, ex_cm_used = aggregate_confusion_from_folds(
        folds, "exercise_confusion_matrix", "exercise_confusion_labels", ex_conf_labels
    )
    form_cm_counts, form_cm_used = aggregate_confusion_from_folds(
        folds, "form_confusion_matrix", "form_confusion_labels", form_conf_labels
    )

    ex_cm_norm = normalize_confusion(ex_cm_counts)
    form_cm_norm = normalize_confusion(form_cm_counts)

    ex_cm_names = [ex_name(x) for x in ex_conf_labels]
    form_cm_names = [form_name(x) for x in form_conf_labels]

    if ex_cm_counts is not None:
        write_matrix_csv(
            out_dir / "cv_confusion_exercise_counts.csv", ex_cm_counts, ex_cm_names
        )
        write_matrix_csv(
            out_dir / "cv_confusion_exercise_norm.csv", ex_cm_norm, ex_cm_names
        )

    if form_cm_counts is not None:
        write_matrix_csv(
            out_dir / "cv_confusion_form_counts.csv", form_cm_counts, form_cm_names
        )
        write_matrix_csv(
            out_dir / "cv_confusion_form_norm.csv", form_cm_norm, form_cm_names
        )

    plots_info = maybe_make_plots(out_dir, fold_rows, ex_summary, form_summary)
    confusion_plots = maybe_make_confusion_plots(
        out_dir,
        ex_cm_counts,
        ex_cm_norm,
        ex_cm_names,
        form_cm_counts,
        form_cm_norm,
        form_cm_names,
    )

    summary = {
        "cv_report_file": str(cv_path.resolve()),
        "dataset_from_report": str(dataset_path),
        "n_samples": int(report.get("n_samples", 0)),
        "cv_folds": int(report.get("cv_folds", len(folds))),
        "splitter": report.get("splitter", ""),
        "seed": report.get("seed", None),
        "train_config": report.get("train_config", {}),
        "aggregate_from_report": report.get("aggregate", {}),
        "metric_stats_recomputed": metric_stats,
        "best_worst_folds": {
            "best_exercise_accuracy_fold": best_ex_fold,
            "worst_exercise_accuracy_fold": worst_ex_fold,
            "best_form_accuracy_fold": best_form_fold,
            "worst_form_accuracy_fold": worst_form_fold,
        },
        "exercise_class_summary": ex_summary,
        "form_class_summary": form_summary,
        "hardest_exercises_by_recall_top3": hardest_ex,
        "hardest_form_by_recall": hardest_form,
        "confusion_summary": {
            "exercise_confusion_folds_used": int(ex_cm_used),
            "exercise_confusion_labels": ex_cm_names,
            "form_confusion_folds_used": int(form_cm_used),
            "form_confusion_labels": form_cm_names,
        },
        "plot_info": {
            "general": plots_info,
            "confusion": confusion_plots,
        },
    }

    summary_path = out_dir / "cv_eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv(
        out_dir / "cv_fold_metrics.csv",
        fold_rows,
        [
            "fold",
            "exercise_accuracy",
            "exercise_f1_macro",
            "exercise_f1_weighted",
            "form_accuracy",
            "form_f1_macro",
            "form_f1_weighted",
            "error_f1_micro",
            "error_f1_macro",
            "best_val_loss",
            "epochs_ran",
            "n_train",
            "n_val",
        ],
    )
    write_csv(
        out_dir / "cv_exercise_per_class_by_fold.csv",
        ex_rows,
        [
            "fold",
            "class_key",
            "class_name",
            "precision",
            "recall",
            "f1_score",
            "support",
        ],
    )
    write_csv(
        out_dir / "cv_form_per_class_by_fold.csv",
        form_rows,
        [
            "fold",
            "class_key",
            "class_name",
            "precision",
            "recall",
            "f1_score",
            "support",
        ],
    )
    write_csv(
        out_dir / "cv_exercise_per_class_summary.csv",
        ex_summary,
        [
            "class_name",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "support_mean",
            "support_std",
        ],
    )
    write_csv(
        out_dir / "cv_form_per_class_summary.csv",
        form_summary,
        [
            "class_name",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "support_mean",
            "support_std",
        ],
    )

    print(f"Saved: {summary_path}")
    print(f"Saved: {out_dir / 'cv_fold_metrics.csv'}")
    print(f"Saved: {out_dir / 'cv_exercise_per_class_by_fold.csv'}")
    print(f"Saved: {out_dir / 'cv_form_per_class_by_fold.csv'}")
    print(f"Saved: {out_dir / 'cv_exercise_per_class_summary.csv'}")
    print(f"Saved: {out_dir / 'cv_form_per_class_summary.csv'}")

    if ex_cm_counts is not None:
        print(f"Saved: {out_dir / 'cv_confusion_exercise_counts.csv'}")
        print(f"Saved: {out_dir / 'cv_confusion_exercise_norm.csv'}")
    if form_cm_counts is not None:
        print(f"Saved: {out_dir / 'cv_confusion_form_counts.csv'}")
        print(f"Saved: {out_dir / 'cv_confusion_form_norm.csv'}")

    if plots_info.get("plots_created"):
        for pth in plots_info.get("files", []):
            print(f"Saved: {pth}")
    else:
        print(f"General plots skipped: {plots_info.get('reason')}")

    if confusion_plots.get("plots_created"):
        for pth in confusion_plots.get("files", []):
            print(f"Saved: {pth}")
    else:
        print(f"Confusion plots skipped: {confusion_plots.get('reason')}")


if __name__ == "__main__":
    main()
