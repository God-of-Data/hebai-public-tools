"""Binoculars with DictaLM 2.0 on a Modal GPU (owner, 2026-10-15). Encrypted input (bino-input branch) is decrypted only
inside the Modal container with REQ_AGE_KEY, passed for this run only (ephemeral secret, not stored in Modal). Scores are
encrypted to recipient.txt inside the container before they leave it. Nothing readable is printed.
score = performer log-perplexity / cross-perplexity(observer, performer)   (Hans et al. 2024); low = machine-like.
usage (GitHub Actions): modal run modal_jobs/bino.py --limit 5000"""
import os
import modal

app = modal.App("hebai-bino")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate==1.1.1", "bitsandbytes==0.44.1", "pyrage==1.2.3",
                      "sentencepiece", "protobuf"))
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})


def pick(rows, limit):
    """stratified subset: every chatbot set and human set capped; held-out training humans/AI for calibration"""
    import collections, random
    r = random.Random(7); by = collections.defaultdict(list)
    for x in rows:
        p = x["id"].split("|")
        by[("dev", p[1]) if p[0] == "dev" else (p[0], p[1])].append(x)
    cap = {("dev", "h"): int(limit * 0.16), ("dev", "a"): int(limit * 0.10)}
    per = max(60, int(limit * 0.74 / max(1, len([k for k in by if k[0] != "dev"]))))
    out = []
    for k, v in by.items():
        r.shuffle(v); out += v[:cap.get(k, per)]
    return out


@app.function(gpu="T4", image=image, timeout=80 * 60, secrets=[secret], memory=24576)
def run(blob: bytes, recipient: str, limit: int) -> bytes:
    import gzip, json, time
    import torch, torch.nn.functional as F
    from pyrage import decrypt, encrypt, x25519
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    rows = [json.loads(l) for l in gzip.decompress(decrypt(blob, [key])).decode().splitlines() if l.strip()]
    rows = pick(rows, limit); print("texts", len(rows), flush=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    t0 = time.time()
    obs = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0", quantization_config=q, device_map="cuda:0").eval()
    perf = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0-instruct", quantization_config=q, device_map="cuda:0").eval()
    print(f"models loaded {time.time() - t0:.0f}s", flush=True)
    out, B = [], 16
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    with torch.no_grad():
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            enc = tok([rows[i]["text"] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda:0")
            lo = obs(**enc).logits[:, :-1].float(); lp = perf(**enc).logits[:, :-1].float()
            tgt = enc.input_ids[:, 1:]; m = enc.attention_mask[:, 1:].float(); n = m.sum(1).clamp(min=1)
            nll_p = (F.cross_entropy(lp.transpose(1, 2), tgt, reduction="none") * m).sum(1) / n
            nll_o = (F.cross_entropy(lo.transpose(1, 2), tgt, reduction="none") * m).sum(1) / n
            xent = ((-(F.softmax(lo, -1) * F.log_softmax(lp, -1)).sum(-1)) * m).sum(1) / n
            for j, i in enumerate(idx):
                out.append({"id": rows[i]["id"], "ntok": int(n[j]), "logppl_perf": round(float(nll_p[j]), 5),
                            "logppl_obs": round(float(nll_o[j]), 5), "xppl": round(float(xent[j]), 5),
                            "bino": round(float(nll_p[j] / xent[j]), 5)})
            if (s // B) % 50 == 0:
                print(f"{len(out)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    rcp = x25519.Recipient.from_str([w for w in recipient.split() if w.startswith("age1")][0])
    return encrypt("\n".join(json.dumps(r) for r in out).encode(), [rcp])


@app.local_entrypoint()
def main(limit: int = 5000):
    blob = open("bino/windows.jsonl.gz.age", "rb").read()
    enc = run.remote(blob, open("recipient.txt").read(), limit)
    os.makedirs("out", exist_ok=True)
    open("out/bino_scores.jsonl.age", "wb").write(enc)
    print("saved encrypted scores", len(enc), "bytes")
