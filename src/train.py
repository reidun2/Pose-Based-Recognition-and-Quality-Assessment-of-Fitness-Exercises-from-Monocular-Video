import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------- Dataset ----------------
class NPZDataset(Dataset):
    def __init__(self, X, y_ex, y_form, y_err=None):
        self.X = torch.tensor(X, dtype=torch.float32)       # (N,T,F)
        self.y_ex = torch.tensor(y_ex, dtype=torch.long)    # (N,)
        self.y_form = torch.tensor(y_form, dtype=torch.long)# (N,)
        self.y_err = None
        if y_err is not None:
            self.y_err = torch.tensor(y_err, dtype=torch.float32)  # (N,E) multi-hot

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.y_err is None:
            return self.X[i], self.y_ex[i], self.y_form[i]
        return self.X[i], self.y_ex[i], self.y_form[i], self.y_err[i]


# ---------------- Model ----------------
class MultiTaskGRU(nn.Module):
    """
    BiGRU encoder -> pooled embedding -> heads:
      - exercise (n_ex)
      - form (2)
      - errors (n_err) optional
    """
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
        # x: (B,T,F)
        h, _ = self.gru(x)            # (B,T,H*)
        emb = h.mean(dim=1)           # simple temporal average pooling (stable baseline)
        emb = self.dropout(self.norm(emb))

        ex_logits = self.ex_head(emb)
        form_logits = self.form_head(emb)

        err_logits = None
        if self.err_head is not None:
            err_logits = self.err_head(emb)  # (B,E), raw logits for BCEWithLogits
        return ex_logits, form_logits, err_logits


def main():
    data_path = BASE_DIR / "data" / "processed" / "samples.npz"
    data = np.load(data_path, allow_pickle=True)

    X = data["X"]
    y_ex = data["y_ex"]
    y_form = data["y_form"]
    ex_map = data["ex_map"].item()

    # optional errors
    y_err = data["y_err"] if "y_err" in data.files else None
    err_map = data["err_map"].item() if "err_map" in data.files else None

    n_ex = len(ex_map)
    F = X.shape[-1]
    n_err = int(y_err.shape[1]) if y_err is not None else 0

    print("Loaded:", data_path)
    print("X:", X.shape, "y_ex:", y_ex.shape, "y_form:", y_form.shape, "n_ex:", n_ex, "F:", F, "n_err:", n_err)

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y_ex if len(np.unique(y_ex)) > 1 else None
    )

    ds_tr = NPZDataset(X[tr_idx], y_ex[tr_idx], y_form[tr_idx], y_err[tr_idx] if y_err is not None else None)
    ds_te = NPZDataset(X[te_idx], y_ex[te_idx], y_form[te_idx], y_err[te_idx] if y_err is not None else None)

    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskGRU(F=F, n_ex=n_ex, n_err=n_err, hidden=128, num_layers=2, dropout=0.2, bidir=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss() if n_err > 0 else None

    # weights for multitask
    w_ex = 1.0
    w_form = 1.0
    w_err = 0.6   # можно поднять до 1.0, если ошибки важнее

    best_val = None
    patience = 6
    bad = 0
    epochs = 30

    for epoch in range(1, epochs + 1):
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

            loss = w_ex * ce(ex_logits, yb_ex) + w_form * ce(form_logits, yb_form)
            if n_err > 0 and err_logits is not None:
                loss = loss + w_err * bce(err_logits, yb_err)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item() * xb.size(0)

        tr_loss = total / len(ds_tr)

        # ---- validation loss ----
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
                vloss = w_ex * ce(ex_logits, yb_ex) + w_form * ce(form_logits, yb_form)
                if n_err > 0 and err_logits is not None:
                    vloss = vloss + w_err * bce(err_logits, yb_err)

                vtotal += vloss.item() * xb.size(0)

        val_loss = vtotal / len(ds_te)
        print(f"Epoch {epoch:02d} | train={tr_loss:.4f} | val={val_loss:.4f}")

        if best_val is None or val_loss < best_val - 1e-4:
            best_val = val_loss
            bad = 0
            # save best
            out = {
                "model": model.state_dict(),
                "ex_map": ex_map,
                "err_map": err_map,
                "F": int(F),
                "n_ex": int(n_ex),
                "n_err": int(n_err),
                "config": {
                    "arch": "MultiTaskGRU",
                    "hidden": 128,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "bidir": True,
                }
            }
            out_path = BASE_DIR / "data" / "processed" / "model.pt"
            torch.save(out, out_path)
            print("Saved best ->", out_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # ---- final report ----
    ckpt = torch.load(BASE_DIR / "data" / "processed" / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_ex_p, all_ex_t = [], []
    all_f_p, all_f_t = [], []

    with torch.no_grad():
        for batch in dl_te:
            if n_err > 0:
                xb, yb_ex, yb_form, _ = batch
            else:
                xb, yb_ex, yb_form = batch
            xb = xb.to(device)

            ex_logits, form_logits, _ = model(xb)
            all_ex_p.append(ex_logits.argmax(dim=1).cpu().numpy())
            all_ex_t.append(yb_ex.numpy())
            all_f_p.append(form_logits.argmax(dim=1).cpu().numpy())
            all_f_t.append(yb_form.numpy())

    ex_p = np.concatenate(all_ex_p); ex_t = np.concatenate(all_ex_t)
    f_p  = np.concatenate(all_f_p);  f_t  = np.concatenate(all_f_t)

    print("\n=== Exercise classification ===")
    print(classification_report(ex_t, ex_p))

    print("\n=== Form (correct/incorrect) ===")
    print(classification_report(f_t, f_p))


if __name__ == "__main__":
    main()
