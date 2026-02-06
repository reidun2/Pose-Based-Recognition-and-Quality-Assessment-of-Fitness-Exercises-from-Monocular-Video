import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence
from pose_smooth import smooth_sequence
from features import landmarks_to_features, pad_or_trim, nan_to_num

BASE_DIR = Path(__file__).resolve().parent.parent


class MultiTaskGRU(nn.Module):
    def __init__(self, F, n_ex, n_err=0, hidden=128, num_layers=2, dropout=0.2, bidir=True):
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

        self.err_head = None
        if self.n_err > 0:
            self.err_head = nn.Linear(out_dim, self.n_err)

    def forward(self, x):
        h, _ = self.gru(x)
        emb = h.mean(dim=1)
        emb = self.dropout(self.norm(emb))
        ex_logits = self.ex_head(emb)
        form_logits = self.form_head(emb)
        err_logits = self.err_head(emb) if self.err_head is not None else None
        return ex_logits, form_logits, err_logits


def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to mp4")
    p.add_argument("--model", default=str(BASE_DIR / "data" / "processed" / "model.pt"))
    p.add_argument("--T", type=int, default=120)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--topk_errors", type=int, default=5)
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")

    ex_map = ckpt["ex_map"]
    id_to_ex = {v: k for k, v in ex_map.items()}
    n_ex = int(ckpt.get("n_ex", len(ex_map)))

    err_map = ckpt.get("err_map", None)
    if isinstance(err_map, dict):
        id_to_err = {v: k for k, v in err_map.items()}
        n_err = int(ckpt.get("n_err", len(err_map)))
    else:
        id_to_err = None
        n_err = int(ckpt.get("n_err", 0))

    F = int(ckpt.get("F", 8))

    # 1) pose
    seq = extract_landmarks_from_video(args.video, max_frames=args.max_frames, stride=args.stride)
    seq = smooth_sequence(seq, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
    seq = normalize_sequence(seq, rotate=True)

    # 2) features
    feats = landmarks_to_features(seq)          # (t,F)
    feats = pad_or_trim(feats, T=args.T)        # (T,F)
    feats = nan_to_num(feats)

    X = torch.tensor(feats[None, ...], dtype=torch.float32)  # (1,T,F)

    # 3) model
    cfg = ckpt.get("config", {})
    model = MultiTaskGRU(
        F=F,
        n_ex=n_ex,
        n_err=n_err,
        hidden=int(cfg.get("hidden", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        bidir=bool(cfg.get("bidir", True)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        ex_logits, form_logits, err_logits = model(X)
        ex_logits = ex_logits.numpy().reshape(-1)
        form_logits = form_logits.numpy().reshape(-1)
        err_logits = err_logits.numpy().reshape(-1) if err_logits is not None else None

    ex_probs = softmax_np(ex_logits)
    form_probs = softmax_np(form_logits)

    ex_id = int(ex_probs.argmax())
    form_id = int(form_probs.argmax())
    ex_name = id_to_ex.get(ex_id, str(ex_id))
    form_name = "correct" if form_id == 1 else "incorrect"

    print("\n=== Prediction ===")
    print("Exercise:", ex_name, f"(p={ex_probs[ex_id]:.3f})")
    print("Form:", form_name, f"(p={form_probs[form_id]:.3f})")
    print("\nExercise probs:", {id_to_ex[i]: float(ex_probs[i]) for i in range(len(ex_probs))})
    print("Form probs:", {"incorrect": float(form_probs[0]), "correct": float(form_probs[1])})

    # optional errors
    if err_logits is not None and id_to_err is not None:
        err_probs = sigmoid_np(err_logits)  # multi-label probabilities
        topk = int(min(args.topk_errors, len(err_probs)))
        top_ids = np.argsort(-err_probs)[:topk]
        print("\n=== Errors (multi-label) top ===")
        for i in top_ids:
            print(f"- {id_to_err.get(int(i), str(int(i)))}: p={float(err_probs[i]):.3f}")


if __name__ == "__main__":
    main()
