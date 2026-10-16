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
ap.add_argument("--fp16", action="store_true", help="autocast fp16 on GPU (arm B, base model at 512 tokens)")
ap.add_argument("--bf16", action="store_true", help="autocast bf16 (NeoDictaBERT recipe, Fable #49)")
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--pool", default="auto", help="auto | mean (mean over non-pad tokens; encoder + linear head)")
ap.add_argument("--no-bucket", action="store_true", help="disable length-bucketed batches (owner 2026-10-16: speed)")
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
NEO = "neo" in a.model.lower()   # NeoDictaBERT (Fable #48 pilot): custom NeoBERT code on the Hub
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=NEO)
HEAD = "seq-cls"
try:
    if a.pool == "mean":
        raise RuntimeError("mean pooling requested")
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, trust_remote_code=NEO)
except Exception as e:   # encoder + linear head (first token, or mean over non-pad tokens with --pool mean)
    HEAD = "wrap-" + ("mean" if a.pool == "mean" else "first-token")
    print("wrapping the encoder:", HEAD, repr(e)[:200], flush=True)
    from transformers import AutoModel
    class _Out:
        def __init__(self, logits): self.logits = logits
    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.enc = AutoModel.from_pretrained(a.model, trust_remote_code=True)
            hs = getattr(self.enc.config, "hidden_size", None) or getattr(self.enc.config, "dim", 768)
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hs, 2))
        def forward(self, **kw):
            o = self.enc(**kw); h = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, (tuple, list)) else o)
            if a.pool == "mean":
                m = kw["attention_mask"].unsqueeze(-1).to(h.dtype); return _Out(self.head((h * m).sum(1) / m.sum(1).clamp(min=1)))
            return _Out(self.head(h[:, 0]))
        def save_pretrained(self, d):
            self.enc.save_pretrained(d); torch.save(self.head.state_dict(), os.path.join(d, "cls_head.pt"))
    model = Wrap()
_fwd_keys = None
def _enc_args(enc):   # drop inputs the encoder's forward doesn't take (e.g. token_type_ids for NeoBERT)
    global _fwd_keys
    if NEO:
        enc.pop("token_type_ids", None)
    return enc
model.to(DEV); print("params", sum(p.numel() for p in model.parameters()), "device", DEV, flush=True)
cvd = []
if a.variant == "cv" or a.variant.startswith("eq_"):   # eq_<fit> (Fable #46): equal-volume LOO / control fine-tunes
    cvd = [r for r in train if r.get("cvdev")]; train = [r for r in train if not r.get("cvdev")]
    if a.variant.startswith("eq_"):
        train = [r for r in train if a.variant in (r.get("eq") or [])]
    print("cv: train", len(train), "held-out", len(cvd), flush=True)
# 5% of split groups held out to watch training (never the eval sets)
groups = sorted({r["group"] for r in train}); random.shuffle(groups); dev_g = set(groups[: len(groups) // 20])
tr = [r for r in train if r["group"] not in dev_g]; dev = [r for r in train if r["group"] in dev_g]
pos = sum(r["label"] for r in tr); w = torch.tensor([1.0, (len(tr) - pos) / max(1, pos)])
lossf = torch.nn.CrossEntropyLoss(weight=w.to(DEV))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
AC_DT = torch.bfloat16 if a.bf16 else torch.float16
_big = DEV == "cuda" and torch.cuda.get_device_properties(0).total_memory > 40e9   # H100 80GB: no micro-batching needed
B = 32; MB = 8 if (NEO and not _big) else 32; steps = int(a.epochs * len(tr) / B)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    if shuffle and not a.no_bucket:   # length buckets: sort inside chunks of 50 batches, then shuffle the batch order
        ch = bs * 50; out = []
        for i in range(0, len(idx), ch):
            c = sorted(idx[i:i + ch], key=lambda j: len(rows[j]["text"])); out += [c[k:k + bs] for k in range(0, len(c), bs)]
        random.shuffle(out)
        for b in out: yield [rows[j] for j in b]
        return
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]

def predict(rows):   # scored in length order (less padding), returned in input order
    order = sorted(range(len(rows)), key=lambda j: len(rows[j]["text"])); res = [0.0] * len(rows)
    for j, p in zip(order, _predict([rows[j] for j in order])): res[j] = p
    return res
def _predict(rows):
    model.eval(); out = []
    with torch.no_grad():
        for b in batches(rows, 16 if NEO else 64, False):
            enc = tok([r["text"] for r in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=AC_DT, enabled=(a.fp16 or a.bf16) and DEV == "cuda"):
                lg = model(**_enc_args(enc)).logits.float()
            out += torch.softmax(lg, -1)[:, 1].cpu().tolist()
    model.train(); return out

def auroc(y, s):
    y = np.array(y); s = np.array(s); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))

scaler = torch.cuda.amp.GradScaler(enabled=a.fp16 and not a.bf16 and DEV == "cuda")
model.train(); step = 0; stop = False
while step < steps and not stop:
    for b in batches(tr, B, True):
        for k in range(0, len(b), MB):   # micro-batches (NeoDictaBERT: 8 x 4 = the same 32 per optimizer step)
            mb = b[k:k + MB]
            enc = tok([r["text"] for r in mb], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=AC_DT, enabled=(a.fp16 or a.bf16) and DEV == "cuda"):
                loss = lossf(model(**_enc_args(enc)).logits.float(), torch.tensor([r["label"] for r in mb], device=DEV)) * len(mb) / len(b)
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); opt.zero_grad(); step += 1
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
    md = f"/tmp/model_{a.variant}"; (model.to("cpu").half() if (a.fp16 or a.bf16) else model.to("cpu")).save_pretrained(md); tok.save_pretrained(md)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(md, arcname=f"model_{a.variant}")
    open(f"out_model/ft_model_{a.variant}.tar.gz.age" if os.path.isdir("out_model") else f"ft_model_{a.variant}.tar.gz.age", "wb").write(encrypt(buf.getvalue(), [rcp]))
import hashlib as _h   # Fable #35: fingerprint so build_second refuses dev scores from another model or export
json.dump({"variant": a.variant, "steps": step, "minutes": round((time.time() - t0) / 60, 1), "tag": os.environ.get("FT_TAG", ""),
           "model": a.model, "max_len": a.max_len, "head": HEAD, "lr": a.lr, "precision": "bf16" if a.bf16 else "fp16" if a.fp16 else "fp32",
           "bucketed": not a.no_bucket,
           "train_sha1": _h.sha1(gzip.decompress(decrypt(open("ft/train.jsonl.gz.age", "rb").read(), [key]))).hexdigest()},
          open(f"out/ft_status_{a.variant}.json", "w"))
