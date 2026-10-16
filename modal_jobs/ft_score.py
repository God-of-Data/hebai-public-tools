"""Score PUBLIC test texts with a fine-tune kept (REQ-key encrypted) in the Modal volume "hebai-models" (arm B: a base
model at 512 tokens is too slow on the 2-CPU workspace). Input texts are age-encrypted to the REQ key (branch
ftscore-input, score/texts.jsonl.age), decrypted only inside the container; the logit cache leaves encrypted to
recipient.txt. The owner's cases and the owner-provided WhatsApp file are never sent here (scored locally).
usage (GitHub Actions): modal run modal_jobs/ft_score.py --tag v37b --max-len 512"""
import io, os, tarfile
import modal

app = modal.App("hebai-ftscore")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.46.3", "pyrage==1.2.3", "numpy<2")
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})
vol = modal.Volume.from_name("hebai-models", create_if_missing=True)


@app.function(gpu="A10G", image=image, timeout=2 * 60 * 60, secrets=[secret], volumes={"/vol": vol})
def score(blob: bytes, recipient: str, tag: str, max_len: int, variant: str = "cv") -> bytes:
    import hashlib, json, tempfile, time
    import torch
    from pyrage import decrypt, encrypt, x25519
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    d = tempfile.mkdtemp()
    tarfile.open(fileobj=io.BytesIO(decrypt(open(f"/vol/ft_model_{variant}_{tag}.tar.gz.age", "rb").read(), [key]))).extractall(d)
    md = f"{d}/model_{variant}"
    texts = [json.loads(l) for l in decrypt(blob, [key]).decode().splitlines() if l.strip()]
    neo = os.path.exists(f"{md}/cls_head.pt")   # Fable #49 Neo recipe: encoder + mean-pooled linear head
    tok = AutoTokenizer.from_pretrained(md, trust_remote_code=neo)
    if neo:
        from transformers import AutoModel
        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.enc = AutoModel.from_pretrained(md, trust_remote_code=True)
                self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(self.enc.config.hidden_size, 2))
                self.head.load_state_dict(torch.load(f"{md}/cls_head.pt", map_location="cpu"))
            def forward(self, **kw):
                kw.pop("token_type_ids", None); h = self.enc(**kw).last_hidden_state; m = kw["attention_mask"].unsqueeze(-1).to(h.dtype)
                class O: pass
                o = O(); o.logits = self.head((h * m).sum(1) / m.sum(1).clamp(min=1)); return o
        model = Wrap().float().cuda().eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(md).float().cuda().eval()
    out, t0 = {}, time.time()
    texts.sort(key=len)
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = texts[i:i + 64]
            enc = tok(b, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16 if neo else torch.float16):
                lg = model(**enc).logits.float()
            for t, v in zip(b, (lg[:, 1] - lg[:, 0]).tolist()):
                out[hashlib.sha1(t.encode("utf-8")).hexdigest()] = round(max(-13.8155, min(13.8155, v)), 5)
    print("scored", len(out), f"{time.time() - t0:.0f}s", flush=True)
    return encrypt(json.dumps(out).encode(), [x25519.Recipient.from_str(recipient)])


@app.local_entrypoint()
def main(tag: str, max_len: int = 512, variant: str = "cv"):
    rcp = [w for w in open("recipient.txt").read().split() if w.startswith("age1")][0]
    blob = score.remote(open("score/texts.jsonl.age", "rb").read(), rcp, tag, max_len, variant)
    os.makedirs("out", exist_ok=True); open("out/scores.json.age", "wb").write(blob); print("saved out/scores.json.age")
