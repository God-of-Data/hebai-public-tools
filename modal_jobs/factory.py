"""Family factory (Fable #22): open-weight chat models write answers to the SAME request sets on a Modal GPU (vLLM).
Encrypted request files (*.txt.age) are decrypted only inside the container with REQ_AGE_KEY (passed for this run only);
answers are encrypted to recipient.txt inside the container. Decoding (pre-registered): each model's own chat template;
70% of requests at temperature 0.7, 30% at 1.0 (fixed by request hash); top_p 0.95; max 700 new tokens.
usage (GitHub Actions): modal run modal_jobs/factory.py --family llama"""
import hashlib, os
import modal

FAMILIES = {   # family: (HF model id, GPU)
    "dictalm": ("dicta-il/dictalm2.0-instruct", "L4"),
    "llama": ("unsloth/Llama-3.1-8B-Instruct", "L4"),
    "gemma": ("unsloth/gemma-3-12b-it", "L40S"),
    "mistral": ("mistralai/Mistral-Nemo-Instruct-2407", "L40S"),
    "aya": ("CohereLabs/aya-expanse-8b", "L4"),
    "phi": ("microsoft/Phi-4-mini-instruct", "L4"),
}
SETS = ["own150.txt", "train300.txt", "wa_test50.txt", "wa_train100.txt", "wa2_test100.txt", "qa_pairs_train.txt.age",
        "qa_pairs_test.txt.age", "qa_casual_train.txt.age", "qa_casual_test.txt.age", "formal_notices120.txt",
        "wa_formatted120.txt", "analytic160.txt", "wa_conv_train300.txt", "wa_conv120.txt"]

app = modal.App("hebai-factory")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm==0.8.5", "pyrage==1.2.3", "transformers==4.51.3"))
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})


def gen(model_id: str, files: dict, recipient: str, family: str) -> bytes:
    import json, time
    from pyrage import decrypt, encrypt, x25519
    from vllm import LLM, SamplingParams
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    reqs = []
    for name, blob in files.items():
        txt = decrypt(blob, [key]).decode() if name.endswith(".age") else blob.decode()
        for i, line in enumerate(l for l in txt.split("\n") if l.strip()):
            reqs.append((name.replace(".txt.age", "").replace(".txt", ""), i, line.strip()))
    print("requests", len(reqs), flush=True)
    t0 = time.time()
    llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.90, trust_remote_code=True)
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)
    temp = lambda s, i: 1.0 if int(hashlib.sha1(f"{s}|{i}".encode()).hexdigest(), 16) % 10 >= 7 else 0.7
    sps = [SamplingParams(temperature=temp(s, i), top_p=0.95, max_tokens=700, seed=i) for s, i, _ in reqs]
    outs = llm.chat([[{"role": "user", "content": q}] for _, _, q in reqs], sps, use_tqdm=False)
    rows = []
    for (s, i, _), o in zip(reqs, outs):
        c = o.outputs[0]
        rows.append({"family": family, "model": model_id, "set": s, "i": i, "temp": temp(s, i),
                     "finish": c.finish_reason, "answer": c.text})
    print(f"generated {len(rows)} in {time.time() - t0:.0f}s; truncated {sum(r['finish'] == 'length' for r in rows)}", flush=True)
    rcp = x25519.Recipient.from_str([w for w in recipient.split() if w.startswith("age1")][0])
    return encrypt("\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode(), [rcp])


fns = {g: app.function(gpu=g, image=image, timeout=75 * 60, secrets=[secret], memory=32768, serialized=True,
                       name=f"gen_{g.lower()}")(gen) for g in sorted({v[1] for v in FAMILIES.values()})}


@app.local_entrypoint()
def main(family: str):
    model_id, gpu = FAMILIES[family]
    files = {n: open(f"requests/{n}", "rb").read() for n in SETS if os.path.exists(f"requests/{n}")}
    enc = fns[gpu].remote(model_id, files, open("recipient.txt").read(), family)
    os.makedirs("out", exist_ok=True)
    open(f"out/factory_{family}.jsonl.age", "wb").write(enc)
    print("saved encrypted answers", len(enc), "bytes")
