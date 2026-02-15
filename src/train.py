import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent


class NPZDataset(Dataset):
    def __init__(self, X, y_ex, y_form, y_err=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_ex = torch.tensor(y_ex, dtype=torch.long)
        self.y_form = torch.tensor(y_form, dtype=torch.long)
        self.y_err = None
        if y_err is not None:
            self.y_err = torch.tensor(y_err, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.y_err is None:
            return self.X[i], self.y_ex[i], self.y_form[i]
        return self.X[i], self.y_ex[i], self.y_form[i], self.y_err[i]


class MultiTaskGRU(nn.Module):
    def __init__(
        self, F, n_ex, n_err=0, hidden=128, num_layers=2, dropout=0.2, bidir=True
    ):
        super().__init__()
        self.n_err = int(n_err)
        self.gru = nn.GRU(
            input_size=F,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ex_head = nn.Linear(out_dim, n_ex)
        self.form_head = nn.Linear(out_dim, 2)
        self.err_head = nn.Linear(out_dim, self.n_err) if self.n_err > 0 else None

    def forward(self, x):
        h, _ = self.gru(x)
        emb = h.mean(dim=1)
        emb = self.dropout(self.norm(emb))
        ex_logits = self.ex_head(emb)
        form_logits = self.form_head(emb)
        err_logits = self.err_head(emb) if self.err_head is not None else None
        return ex_logits, form_logits, err_logits


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_checkpoint(state_dict, ex_map, err_map, F, n_ex, n_err, T, cfg):
    return {
        "model": state_dict,
        "ex_map": ex_map,
        "err_map": err_map,
        "F": int(F),
        "n_ex": int(n_ex),
        "n_err": int(n_err),
        "T": int(T),
        "config": {
            "arch": "MultiTaskGRU",
            "hidden": int(cfg["hidden"]),
            "num_layers": int(cfg["num_layers"]),
            "dropout": float(cfg["dropout"]),
            "bidir": bool(cfg["bidir"]),
        },
    }


def evaluate_split(model, dl_te, device, n_err, n_ex):
    all_ex_p, all_ex_t = [], []
    all_f_p, all_f_t = [], []
    all_err_t, all_err_p = [], []

    model.eval()
    with torch.no_grad():
        for batch in dl_te:
            if n_err > 0:
                xb, yb_ex, yb_form, yb_err = batch
            else:
                xb, yb_ex, yb_form = batch
                yb_err = None

            xb = xb.to(device)
            ex_logits, form_logits, err_logits = model(xb)

            all_ex_p.append(ex_logits.argmax(dim=1).cpu().numpy())
            all_ex_t.append(yb_ex.numpy())
            all_f_p.append(form_logits.argmax(dim=1).cpu().numpy())
            all_f_t.append(yb_form.numpy())

            if yb_err is not None and err_logits is not None:
                probs = torch.sigmoid(err_logits).cpu().numpy()
                all_err_p.append((probs >= 0.5).astype(np.int32))
                all_err_t.append(yb_err.numpy().astype(np.int32))

    ex_p = np.concatenate(all_ex_p)
    ex_t = np.concatenate(all_ex_t)
    f_p = np.concatenate(all_f_p)
    f_t = np.concatenate(all_f_t)

    ex_acc = float(np.mean(ex_p == ex_t))
    form_acc = float(np.mean(f_p == f_t))
    ex_report_dict = classification_report(
        ex_t, ex_p, zero_division=0, output_dict=True
    )
    form_report_dict = classification_report(
        f_t, f_p, zero_division=0, output_dict=True
    )
    ex_report_text = classification_report(ex_t, ex_p, zero_division=0)
    form_report_text = classification_report(f_t, f_p, zero_division=0)
    ex_labels = list(range(int(n_ex)))
    form_labels = [0, 1]
    ex_cm = confusion_matrix(ex_t, ex_p, labels=ex_labels)
    form_cm = confusion_matrix(f_t, f_p, labels=form_labels)

    out = {
        "exercise_accuracy": ex_acc,
        "form_accuracy": form_acc,
        "exercise_report": ex_report_dict,
        "form_report": form_report_dict,
        "exercise_report_text": ex_report_text,
        "form_report_text": form_report_text,
        "exercise_confusion_labels": ex_labels,
        "exercise_confusion_matrix": ex_cm.astype(int).tolist(),
        "form_confusion_labels": form_labels,
        "form_confusion_matrix": form_cm.astype(int).tolist(),
    }

    if all_err_t:
        err_t = np.concatenate(all_err_t, axis=0)
        err_p = np.concatenate(all_err_p, axis=0)
        micro = f1_score(err_t, err_p, average="micro", zero_division=0)
        macro = f1_score(err_t, err_p, average="macro", zero_division=0)
        out["error_f1_micro"] = float(micro)
        out["error_f1_macro"] = float(macro)

    return out


def train_one_split(
    X,
    y_ex,
    y_form,
    y_err,
    tr_idx,
    te_idx,
    args,
    ex_map,
    err_map,
    T,
    split_name,
    seed_offset=0,
):
    set_seed(int(args.seed) + int(seed_offset))

    n_ex = len(ex_map)
    F = X.shape[-1]
    n_err = int(y_err.shape[1]) if y_err is not None else 0

    ds_tr = NPZDataset(
        X[tr_idx],
        y_ex[tr_idx],
        y_form[tr_idx],
        y_err[tr_idx] if y_err is not None else None,
    )
    ds_te = NPZDataset(
        X[te_idx],
        y_ex[te_idx],
        y_form[te_idx],
        y_err[te_idx] if y_err is not None else None,
    )

    dl_tr = DataLoader(
        ds_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=False
    )
    dl_te = DataLoader(
        ds_te, batch_size=int(args.val_batch_size), shuffle=False, drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "hidden": int(args.hidden),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "bidir": bool(args.bidir),
    }
    model = MultiTaskGRU(F=F, n_ex=n_ex, n_err=n_err, **cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss() if n_err > 0 else None

    best_val = None
    best_state = None
    bad = 0
    epochs_ran = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0

        for batch in dl_tr:
            if n_err > 0:
                xb, yb_ex, yb_form, yb_err = batch
                yb_err = yb_err.to(device)
            else:
                xb, yb_ex, yb_form = batch
                yb_err = None

            xb = xb.to(device)
            yb_ex = yb_ex.to(device)
            yb_form = yb_form.to(device)

            ex_logits, form_logits, err_logits = model(xb)
            loss = float(args.w_ex) * ce(ex_logits, yb_ex) + float(args.w_form) * ce(
                form_logits, yb_form
            )
            if n_err > 0 and err_logits is not None:
                loss = loss + float(args.w_err) * bce(err_logits, yb_err)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item() * xb.size(0)

        tr_loss = total / max(1, len(ds_tr))

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for batch in dl_te:
                if n_err > 0:
                    xb, yb_ex, yb_form, yb_err = batch
                    yb_err = yb_err.to(device)
                else:
                    xb, yb_ex, yb_form = batch
                    yb_err = None

                xb = xb.to(device)
                yb_ex = yb_ex.to(device)
                yb_form = yb_form.to(device)

                ex_logits, form_logits, err_logits = model(xb)
                vloss = float(args.w_ex) * ce(ex_logits, yb_ex) + float(
                    args.w_form
                ) * ce(form_logits, yb_form)
                if n_err > 0 and err_logits is not None:
                    vloss = vloss + float(args.w_err) * bce(err_logits, yb_err)
                vtotal += vloss.item() * xb.size(0)

        val_loss = vtotal / max(1, len(ds_te))
        epochs_ran = epoch
        print(
            f"[{split_name}] Epoch {epoch:02d} | train={tr_loss:.4f} | val={val_loss:.4f}"
        )

        if best_val is None or val_loss < best_val - 1e-4:
            best_val = val_loss
            bad = 0
            best_state = clone_state_dict(model.state_dict())
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[{split_name}] Early stopping.")
                break

    if best_state is None:
        best_state = clone_state_dict(model.state_dict())
        best_val = float("nan")

    model.load_state_dict(best_state)
    metrics = evaluate_split(model, dl_te, device, n_err, n_ex)
    metrics["best_val_loss"] = float(best_val)
    metrics["epochs_ran"] = int(epochs_ran)
    metrics["n_train"] = int(len(ds_tr))
    metrics["n_val"] = int(len(ds_te))

    ckpt = build_checkpoint(
        state_dict=best_state,
        ex_map=ex_map,
        err_map=err_map,
        F=F,
        n_ex=n_ex,
        n_err=n_err,
        T=T,
        cfg=cfg,
    )
    return metrics, ckpt


def choose_cv_splitter(y_ex, n_splits, seed):
    y_ex = np.asarray(y_ex)
    unique, counts = np.unique(y_ex, return_counts=True)
    min_count = int(np.min(counts)) if len(counts) > 0 else 0

    if len(unique) > 1 and min_count >= int(n_splits):
        splitter = StratifiedKFold(
            n_splits=int(n_splits), shuffle=True, random_state=int(seed)
        )
        return splitter.split(np.arange(len(y_ex)), y_ex), "StratifiedKFold", min_count

    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    return splitter.split(np.arange(len(y_ex))), "KFold", min_count


def aggregate_fold_metrics(fold_metrics):
    if not fold_metrics:
        return {}
    keys = set()
    for fm in fold_metrics:
        keys.update(fm.keys())

    out = {}
    for k in sorted(keys):
        vals = [fm[k] for fm in fold_metrics if isinstance(fm.get(k), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float32)
        out[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", default=str(BASE_DIR / "data" / "processed" / "reps_dataset.npz")
    )
    p.add_argument("--out", default=str(BASE_DIR / "data" / "processed" / "model.pt"))
    p.add_argument(
        "--cv_out", default=str(BASE_DIR / "data" / "processed" / "cv_report.json")
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument(
        "--cv_folds", type=int, default=1, help="Set 5 for 5-fold cross-validation."
    )

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--w_ex", type=float, default=1.0)
    p.add_argument("--w_form", type=float, default=1.0)
    p.add_argument("--w_err", type=float, default=0.6)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidir", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--save_fold_models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When CV is enabled, save each fold model to model_fold{i}.pt",
    )
    args = p.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    data = np.load(data_path, allow_pickle=True)

    X = data["X"]
    y_ex = data["y_ex"]
    if "y_form" in data.files:
        y_form = data["y_form"]
    elif "y_err" in data.files:
        y_form = (np.sum(data["y_err"], axis=1) == 0).astype(np.int64)
    else:
        raise ValueError("Dataset must contain y_form or y_err to derive y_form.")

    ex_map = data["ex_map"].item()
    y_err = data["y_err"] if "y_err" in data.files else None
    err_map = data["err_map"].item() if "err_map" in data.files else None
    T = int(data["T"][0]) if "T" in data.files else int(X.shape[1])

    n_ex = len(ex_map)
    F = X.shape[-1]
    n_err = int(y_err.shape[1]) if y_err is not None else 0

    print("Loaded:", data_path)
    print(
        "X:",
        X.shape,
        "y_ex:",
        y_ex.shape,
        "y_form:",
        y_form.shape,
        "n_ex:",
        n_ex,
        "F:",
        F,
        "n_err:",
        n_err,
    )
    ex_counts = {k: int((y_ex == v).sum()) for k, v in ex_map.items()}
    print("Exercise distribution:", ex_counts)
    print(
        "Form distribution:",
        {"incorrect": int((y_form == 0).sum()), "correct": int((y_form == 1).sum())},
    )
    if y_err is not None and err_map is not None:
        err_counts = {k: int(y_err[:, v].sum()) for k, v in err_map.items()}
        print("Error positives:", err_counts)

    idx = np.arange(len(X))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if int(args.cv_folds) <= 1:
        try:
            tr_idx, te_idx = train_test_split(
                idx,
                test_size=float(args.test_size),
                random_state=int(args.seed),
                stratify=y_ex if len(np.unique(y_ex)) > 1 else None,
            )
        except ValueError:
            tr_idx, te_idx = train_test_split(
                idx,
                test_size=float(args.test_size),
                random_state=int(args.seed),
                stratify=None,
            )

        metrics, ckpt = train_one_split(
            X=X,
            y_ex=y_ex,
            y_form=y_form,
            y_err=y_err,
            tr_idx=tr_idx,
            te_idx=te_idx,
            args=args,
            ex_map=ex_map,
            err_map=err_map,
            T=T,
            split_name="holdout",
            seed_offset=0,
        )

        torch.save(ckpt, out_path)
        print("Saved best ->", out_path)

        print("\n=== Exercise classification ===")
        print(metrics["exercise_report_text"])
        print("\n=== Form (correct/incorrect) ===")
        print(metrics["form_report_text"])
        if "error_f1_micro" in metrics:
            print("\n=== Error multi-label ===")
            print(f"F1 micro@0.5: {metrics['error_f1_micro']:.4f}")
            print(f"F1 macro@0.5: {metrics['error_f1_macro']:.4f}")
        return

    # ---- CV mode ----
    n_splits = int(args.cv_folds)
    if n_splits < 2:
        raise ValueError("--cv_folds must be >= 2 for CV mode.")

    split_iter, splitter_name, min_count = choose_cv_splitter(
        y_ex, n_splits=n_splits, seed=args.seed
    )
    print(
        f"\nCV mode: {n_splits}-fold using {splitter_name} (min class count={min_count})"
    )
    if splitter_name != "StratifiedKFold":
        print(
            "Warning: fallback to KFold (class counts are too small for stratification)."
        )

    fold_reports = []
    fold_ckpts = []
    best_fold_i = None
    best_fold_score = None

    for i, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        split_name = f"fold{i}"
        print(f"\n===== {split_name} ({len(tr_idx)} train / {len(te_idx)} val) =====")
        metrics, ckpt = train_one_split(
            X=X,
            y_ex=y_ex,
            y_form=y_form,
            y_err=y_err,
            tr_idx=np.array(tr_idx, dtype=np.int64),
            te_idx=np.array(te_idx, dtype=np.int64),
            args=args,
            ex_map=ex_map,
            err_map=err_map,
            T=T,
            split_name=split_name,
            seed_offset=i,
        )
        fold_reports.append({"fold": i, **metrics})
        fold_ckpts.append(ckpt)

        score = (
            float(metrics.get("exercise_accuracy", 0.0)),
            -float(metrics.get("best_val_loss", 1e9)),
        )
        if best_fold_score is None or score > best_fold_score:
            best_fold_score = score
            best_fold_i = i - 1

        print(
            f"[{split_name}] ex_acc={metrics['exercise_accuracy']:.4f} | form_acc={metrics['form_accuracy']:.4f}"
        )
        if "error_f1_micro" in metrics:
            print(
                f"[{split_name}] err_f1_micro={metrics['error_f1_micro']:.4f} | err_f1_macro={metrics['error_f1_macro']:.4f}"
            )

        if bool(args.save_fold_models):
            fold_out = out_path.with_name(f"{out_path.stem}_fold{i}{out_path.suffix}")
            torch.save(ckpt, fold_out)
            print(f"Saved fold model -> {fold_out}")

    aggregate = aggregate_fold_metrics(fold_reports)
    cv_report = {
        "data": str(data_path.resolve()),
        "n_samples": int(len(X)),
        "cv_folds": int(n_splits),
        "splitter": splitter_name,
        "seed": int(args.seed),
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "val_batch_size": int(args.val_batch_size),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "w_ex": float(args.w_ex),
            "w_form": float(args.w_form),
            "w_err": float(args.w_err),
            "hidden": int(args.hidden),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "bidir": bool(args.bidir),
        },
        "folds": to_builtin(fold_reports),
        "aggregate": to_builtin(aggregate),
    }

    cv_out = Path(args.cv_out)
    cv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_out, "w", encoding="utf-8") as f:
        json.dump(cv_report, f, ensure_ascii=False, indent=2)
    print("\nSaved CV report ->", cv_out)

    if best_fold_i is not None:
        torch.save(fold_ckpts[best_fold_i], out_path)
        print(f"Saved best-fold model (fold {best_fold_i + 1}) -> {out_path}")

    print("\n=== CV aggregate ===")
    for k in sorted(aggregate.keys()):
        v = aggregate[k]
        print(f"{k}: mean={v['mean']:.4f} std={v['std']:.4f}")


if __name__ == "__main__":
    main()
