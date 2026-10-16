"""Gemini paid run (owner, 2026-10-16: prepaid ~15 ILS; "save the raw data"). Key: GEMINI_PAID_API_KEY.
  models : list generateContent models (no content sent)
  probe  : 3 own150 requests per model, synchronous; prints usage (incl. thinking tokens); raw saved encrypted
  submit : Batch API (half price) with inline requests built from our own request files; input saved encrypted
  fetch  : batch state; when done, the RAW responses exactly as returned (every field), encrypted
Nothing readable is written. usage: python gemini_batch.py models | probe <tag> <m1,m2> | submit <tag> <model> <sets> [max_out]
| fetch <tag>"""
import json, os, sys, urllib.request
from pyrage import decrypt, encrypt, x25519

KEY = os.environ["GEMINI_PAID_API_KEY"].strip()
API = "https://generativelanguage.googleapis.com/v1beta"
RCP = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
os.makedirs("out", exist_ok=True)
THINK = os.environ.get("THINKING", "")   # "" = model default; else thinkingLevel (e.g. low)


def call(method, path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"::error::{method} {path.split('?')[0]}: {e.code} {e.read()[:600].decode(errors='replace')}"); sys.exit(1)


def enc(name, b):
    open(f"out/{name}", "wb").write(encrypt(b, [RCP]))


def gen_config(max_out):
    c = {"maxOutputTokens": max_out}
    if THINK: c["thinkingConfig"] = {"thinkingLevel": THINK}
    return c


mode = sys.argv[1]
if mode in ("models",): sys.argv = sys.argv[:2]
if mode == "models":
    ms = [m["name"] for m in call("GET", "/models?pageSize=200")["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]
    print("::notice::paid key OK; models: " + ", ".join(m for m in ms if "gemini" in m)[:3000])
elif mode == "probe":
    qs = [l.strip() for l in open("requests/own150.txt", encoding="utf-8") if l.strip()][:3]
    raws = []
    for m in sys.argv[3].split(","):
        for q in qs:
            r = call("POST", f"/{m}:generateContent", {"contents": [{"role": "user", "parts": [{"text": q}]}], "generationConfig": gen_config(1500)})
            raws.append({"model": m, "response": r}); u = r.get("usageMetadata", {})
            txt = "".join(p.get("text", "") for p in r["candidates"][0]["content"].get("parts", []))
            print(f"::notice::{m}: finish {r['candidates'][0].get('finishReason')} prompt {u.get('promptTokenCount')} "
                  f"out {u.get('candidatesTokenCount')} thinking {u.get('thoughtsTokenCount')} words {len(txt.split())}")
    enc(f"gemini_probe_{sys.argv[2]}.jsonl.age", "\n".join(json.dumps(r, ensure_ascii=False) for r in raws).encode())
elif mode == "submit":
    tag, model, sets = sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    max_out = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    reqs = []
    for s in sets:
        p = f"requests/{s}.txt.age" if os.path.exists(f"requests/{s}.txt.age") else f"requests/{s}.txt"
        blob = open(p, "rb").read()
        txt = decrypt(blob, [key]).decode() if p.endswith(".age") else blob.decode()
        for i, q in enumerate(l.strip() for l in txt.split("\n") if l.strip()):
            reqs.append({"request": {"contents": [{"role": "user", "parts": [{"text": q}]}], "generationConfig": gen_config(max_out)},
                         "metadata": {"key": f"{tag}|{s}|{i}"}})
    enc(f"gemini_input_{tag}.jsonl.age", "\n".join(json.dumps(r, ensure_ascii=False) for r in reqs).encode())
    op = call("POST", f"/{model}:batchGenerateContent",
              {"batch": {"display_name": tag, "input_config": {"requests": {"requests": reqs}}}})
    meta = {"tag": tag, "model": model, "sets": sets, "n": len(reqs), "max_out": max_out, "thinking": THINK, "name": op.get("name")}
    json.dump(meta, open(f"out/gemini_batch_{tag}.json", "w"), indent=1)
    print(f"::notice::gemini batch {tag}: {len(reqs)} requests, {model}, {op.get('name')}")
elif mode == "fetch":
    tag = sys.argv[2]
    meta = json.load(open(f"state/gemini_batch_{tag}.json"))
    op = call("GET", f"/{meta['name']}")
    md = op.get("metadata", {})
    meta.update({"done": op.get("done", False), "state": md.get("state"), "stats": md.get("batchStats")})
    json.dump(meta, open(f"out/gemini_batch_{tag}.json", "w"), indent=1)
    print(f"::notice::gemini batch {tag}: state {md.get('state')} stats {json.dumps(md.get('batchStats'))} done {op.get('done')}")
    if op.get("done"):
        resp = op.get("response", {})
        inl = (resp.get("inlinedResponses") or {}).get("inlinedResponses") or md.get("output", {}).get("inlinedResponses", {}).get("inlinedResponses") or []
        enc(f"gemini_raw_{tag}.jsonl.age", "\n".join(json.dumps(r, ensure_ascii=False) for r in inl).encode())
        enc(f"gemini_op_{tag}.json.age", json.dumps(op, ensure_ascii=False).encode())   # the whole operation, raw
        u = [(r.get("response") or {}).get("usageMetadata", {}) for r in inl]
        print(f"::notice::{tag}: raw saved encrypted, {len(inl)} responses; prompt {sum(x.get('promptTokenCount', 0) for x in u)} "
              f"out {sum(x.get('candidatesTokenCount', 0) for x in u)} thinking {sum(x.get('thoughtsTokenCount', 0) for x in u)}")
