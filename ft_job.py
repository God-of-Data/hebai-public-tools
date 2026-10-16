"""Fine-tune DictaBERT-tiny as an AI-text classifier on the runner's CPU (Fable #11, step 2). Inputs are age-encrypted
(ft/train.jsonl.gz.age, ft/eval.jsonl.gz.age on branch ft-input; decrypted here with secret REQ_AGE_KEY). Variant
"topic" replaces every word outside the 2,000 most frequent TRAINING words with [MASK] in train and eval alike (the
topic-masked control). Output: per-eval-item AI probability, encrypted to recipient.txt -> out/ft_scores_<variant>.jsonl.age
Variant "cv" trains on the 80% of split groups with cvdev == 0 and also scores the held-out 20% of TRAINING records
(out/ft_dev_cv.jsonl.age), used to calibrate the second opinion on training data only (Fable #14).
usage: python ft_job.py --variant full|topic|cv"""
import argparse, collections, gzip, json, os, random, sys, time
import numpy as np
import torch
from pyrage import decrypt, encrypt, x25519
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="full")
ap.add_argument("--model", default="dicta-il/dictabert-tiny")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--max-minutes", type=float, default=300)
a = ap.parse_args()
t0 = time.time()
torch.set_num_threads(os.cpu_count() or 4); torch.manual_seed(0); random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # GPU when run on Modal (modal_jobs/ft.py)
key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
load = lambda p: [json.loads(l) for l in gzip.decompress(decrypt(open(p, "rb").read(), [key])).decode().splitlines() if l.strip()]
train, ev = load("ft/train.jsonl.gz.age"), load("ft/eval.jsonl.gz.age")
print("train", len(train), "ai", sum(r["label"] for r in train), "eval", len(ev), flush=True)
if a.variant == "topic":
    wc = collections.Counter(w for r in train for w in r["text"].split())
    topw = {w for w, _ in wc.most_common(2000)}
    mask = lambda t: " ".join(w if w in topw else "[MASK]" for w in t.split())
    for r in train + ev:
        r["text"] = mask(r["text"])
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
model.to(DEV); print("params", sum(p.numel() for p in model.parameters()), "device", DEV, flush=True)
cvd = []
if a.variant == "cv":
    cvd = [r for r in train if r.get("cvdev")]; train = [r for r in train if not r.get("cvdev")]
    print("cv: train", len(train), "held-out", len(cvd), flush=True)
# 5% of split groups held out to watch training (never the eval sets)
groups = sorted({r["group"] for r in train}); random.shuffle(groups); dev_g = set(groups[: len(groups) // 20])
tr = [r for r in train if r["group"] not in dev_g]; dev = [r for r in train if r["group"] in dev_g]
pos = sum(r["label"] for r in tr); w = torch.tensor([1.0, (len(tr) - pos) / max(1, pos)])
lossf = torch.nn.CrossEntropyLoss(weight=w.to(DEV))
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
B = 32; steps = int(a.epochs * len(tr) / B)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]

def predict(rows):
    model.eval(); out = []
    with torch.no_grad():
        for b in batches(rows, 64, False):
            enc = tok([r["text"] for r in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(DEV)
            out += torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist()
    model.train(); return out

def auroc(y, s):
    y = np.array(y); s = np.array(s); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))

model.train(); step = 0; stop = False
while step < steps and not stop:
    for b in batches(tr, B, True):
        enc = tok([r["text"] for r in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(DEV)
        loss = lossf(model(**enc).logits, torch.tensor([r["label"] for r in b], device=DEV))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step >= steps or (time.time() - t0) / 60 > a.max_minutes - 25:
            stop = (time.time() - t0) / 60 > a.max_minutes - 25; break
dv = predict(dev)
print("::notice::variant", a.variant, "steps", step, "dev AUROC", round(auroc([r["label"] for r in dev], dv), 4), flush=True)
ps = predict(ev)
os.makedirs("out", exist_ok=True)
rcp = x25519.Recipient.from_str(open("recipient.txt").read().split()[0])
blob = "\n".join(json.dumps({"id": r["id"], "p": round(p, 6)}) for r, p in zip(ev, ps)).encode()
open(f"out/ft_scores_{a.variant}.jsonl.age", "wb").write(encrypt(blob, [rcp]))
if cvd:
    pc = predict(cvd)
    blob = "\n".join(json.dumps({"id": r["id"], "p": round(p, 6)}) for r, p in zip(cvd, pc)).encode()
    open("out/ft_dev_cv.jsonl.age", "wb").write(encrypt(blob, [rcp]))
if os.environ.get("SAVE_MODEL"):   # weights + tokenizer, encrypted to recipient.txt (uploaded to a release by the workflow)
    import io, tarfile
    md = f"/tmp/model_{a.variant}"; model.to("cpu").save_pretrained(md); tok.save_pretrained(md)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(md, arcname=f"model_{a.variant}")
    open(f"out_model/ft_model_{a.variant}.tar.gz.age" if os.path.isdir("out_model") else f"ft_model_{a.variant}.tar.gz.age", "wb").write(encrypt(buf.getvalue(), [rcp]))
json.dump({"variant": a.variant, "steps": step, "minutes": round((time.time() - t0) / 60, 1)}, open(f"out/ft_status_{a.variant}.json", "w"))
