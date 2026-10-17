"""Small character CNN fits on Modal GPUs (Fable #80 step 1, pre-registered). Input: cnn/bundle.age from branch cnn-input —
age-encrypted gzip JSON with the texts (detector view, <= 600 chars), labels and job index sets — decrypted only inside
the containers with REQ_AGE_KEY (ephemeral secret for this run). Each container trains one fit and scores that job's
test rows; the scores are encrypted to recipient.txt before they leave the container.
Model (fixed in advance): char embedding 48; Conv1d widths 3 / 5 / 7 x 128 filters, ReLU, masked global max-pool;
dense 384 -> 128 ReLU; logit. Char vocabulary = training chars seen >= 20 times (others -> UNK). Balanced classes
(pos_weight = n_human / n_ai), Adam lr 1e-3, batch 256, 3 epochs, seed 0. usage (GitHub Actions): modal run modal_jobs/cnn.py"""
import os
import modal

app = modal.App("hebai-cnn")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "pyrage==1.2.3", "numpy<2")
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})


@app.function(gpu="L4", image=image, memory=16384, timeout=40 * 60, secrets=[secret])
def fit(bundle: bytes, recipient: str, job: str) -> tuple:
    import collections, gzip, io, json, time
    import numpy as np, torch, torch.nn as nn
    from pyrage import decrypt, encrypt, x25519
    t0 = time.time(); torch.manual_seed(0); np.random.seed(0)
    D = json.loads(gzip.decompress(decrypt(bundle, [x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())])))
    T, y = D["texts"], np.array(D["y"]); tr, te = D["jobs"][job]
    cnt = collections.Counter(c for i in tr for c in T[i])
    vocab = {c: k + 2 for k, c in enumerate(sorted(c for c, n in cnt.items() if n >= 20))}   # 0 pad, 1 UNK
    L = 600
    def enc(idx):
        X = np.zeros((len(idx), L), dtype=np.int64)
        for r, i in enumerate(idx):
            s = [vocab.get(c, 1) for c in T[i][:L]]; X[r, :len(s)] = s
        return torch.from_numpy(X)
    class CNN(nn.Module):
        def __init__(s, V):
            super().__init__(); s.emb = nn.Embedding(V, 48, padding_idx=0)
            s.convs = nn.ModuleList([nn.Conv1d(48, 128, k, padding=k // 2) for k in (3, 5, 7)])
            s.fc = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))
        def forward(s, x):
            m = (x > 0).unsqueeze(1); e = s.emb(x).transpose(1, 2)
            h = [torch.relu(c(e)).masked_fill(~m, -1e4).amax(2) for c in s.convs]
            return s.fc(torch.cat(h, 1)).squeeze(1)
    dev = os.environ.get("CNN_DEV", "cuda"); net = CNN(len(vocab) + 2).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    Xtr, ytr = enc(tr), torch.tensor(y[tr], dtype=torch.float32)
    pw = torch.tensor(float((ytr == 0).sum() / max(1, (ytr == 1).sum())), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for ep in range(3):
        net.train(); perm = torch.randperm(len(tr), generator=g); tot = 0.0
        for b in range(0, len(tr), 256):
            j = perm[b:b + 256]; xb, yb = Xtr[j].to(dev), ytr[j].to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step(); tot += l.item() * len(j)
        print(f"{job} epoch {ep + 1} loss {tot / len(tr):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    net.eval(); out = []
    with torch.no_grad():
        Xte = enc(te)
        for b in range(0, len(te), 1024):
            out.append(net(Xte[b:b + 1024].to(dev)).float().cpu().numpy())
    buf = io.BytesIO(); np.save(buf, np.concatenate(out).astype(np.float64))
    return job, encrypt(buf.getvalue(), [x25519.Recipient.from_str(recipient)]), D["fp"], round((time.time() - t0) / 60, 1), npar, len(vocab)


@app.local_entrypoint()
def main():
    import time
    bundle = open("cnn/bundle.age", "rb").read()
    rcp = [w for w in open("recipient.txt").read().split() if w.startswith("age1")][0]
    names = [l.strip() for l in open("cnn/jobs.txt") if l.strip()]
    os.makedirs("out", exist_ok=True); t0 = time.time()
    for job, blob, fp, mins, npar, nv in fit.starmap([(bundle, rcp, j) for j in names]):
        open(f"out/{job}.npy.age", "wb").write(blob)
        print(f"job {job} done in {mins} min, params {npar}, chars {nv} (t+{(time.time() - t0) / 60:.1f})", flush=True)
    open("out/fp.txt", "w").write(fp + "\n")
    print("saved", sorted(os.listdir("out")), f"total {(time.time() - t0) / 60:.1f} min")
