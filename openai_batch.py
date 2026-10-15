"""OpenAI Batch API job (owner, 2026-10-16: ChatGPT family, $5 project budget; "save the raw data").
  check  : list the models this key sees (no content sent); writes out/openai_models.json
  submit : builds the batch input from our own request files (decrypted in memory), uploads it, starts the batch;
           the input JSONL is saved ENCRYPTED (out/openai_input_<tag>.jsonl.age); batch id -> out/openai_batch_<tag>.json
  fetch  : downloads the RAW output and error files exactly as OpenAI returns them, encrypted
           (out/openai_raw_<tag>.jsonl.age, out/openai_err_<tag>.jsonl.age) + batch metadata (no content)
Nothing readable is written. usage: python openai_batch.py check | submit <tag> <model> <sets,...> [max_tokens] | fetch <tag>"""
import hashlib, io, json, os, sys, time, urllib.request
from pyrage import decrypt, encrypt, x25519

KEY = os.environ["OPENAI_API_KEY"].strip()
API = "https://api.openai.com/v1"
RCP = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
os.makedirs("out", exist_ok=True)


def call(method, path, data=None, ctype="application/json", raw=False):
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {KEY}", **({"Content-Type": ctype} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            b = r.read()
    except urllib.error.HTTPError as e:
        print(f"::error::{method} {path}: {e.code} {e.read()[:600].decode(errors='replace')}"); sys.exit(1)
    return b if raw else json.loads(b)


def enc(name, b):
    open(f"out/{name}", "wb").write(encrypt(b, [RCP]))


mode = sys.argv[1]
if mode == "check": sys.argv = sys.argv[:2]
if mode == "check":
    ids = sorted(m["id"] for m in call("GET", "/models")["data"])
    keep = [i for i in ids if i.startswith(("gpt", "chat", "o"))]
    json.dump(keep, open("out/openai_models.json", "w"), indent=1)
    print("::notice::key OK; models: " + ", ".join(keep)[:3000])

elif mode == "probe":   # 3 own150 requests per model, synchronous; raw responses saved encrypted; prints usage only
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    qs = [l.strip() for l in open("requests/own150.txt", encoding="utf-8") if l.strip()][:3]
    raws = []
    for model in sys.argv[3].split(","):
        for q in qs:
            body = {"model": model, "messages": [{"role": "user", "content": q}], "max_completion_tokens": 1200}
            if os.environ.get("REASONING"): body["reasoning_effort"] = os.environ["REASONING"]
            r = call("POST", "/chat/completions", json.dumps(body).encode())
            raws.append(r); u = r.get("usage", {})
            print(f"::notice::{model}: finish {r['choices'][0]['finish_reason']} prompt {u.get('prompt_tokens')} "
                  f"completion {u.get('completion_tokens')} reasoning {(u.get('completion_tokens_details') or {}).get('reasoning_tokens')} "
                  f"words {len((r['choices'][0]['message'].get('content') or '').split())}")
    enc(f"openai_probe_{sys.argv[2]}.jsonl.age", "\n".join(json.dumps(r, ensure_ascii=False) for r in raws).encode())

elif mode == "submit":
    tag, model, sets = sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    maxtok = int(sys.argv[5]) if len(sys.argv) > 5 else 1200
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    lines = []
    for s in sets:
        p = f"requests/{s}.txt.age" if os.path.exists(f"requests/{s}.txt.age") else f"requests/{s}.txt"
        blob = open(p, "rb").read()
        txt = decrypt(blob, [key]).decode() if p.endswith(".age") else blob.decode()
        for i, q in enumerate(l.strip() for l in txt.split("\n") if l.strip()):
            body = {"model": model, "messages": [{"role": "user", "content": q}], "max_completion_tokens": maxtok}
            if os.environ.get("REASONING"): body["reasoning_effort"] = os.environ["REASONING"]
            lines.append(json.dumps({"custom_id": f"{tag}|{s}|{i}", "method": "POST", "url": "/v1/chat/completions", "body": body},
                                    ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode()
    enc(f"openai_input_{tag}.jsonl.age", data)
    bd = b"----hebai" + hashlib.sha1(data).hexdigest()[:12].encode()
    mp = (b"--" + bd + b'\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n--' + bd +
          b'\r\nContent-Disposition: form-data; name="file"; filename="in.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n' +
          data + b"\r\n--" + bd + b"--\r\n")
    f = call("POST", "/files", mp, "multipart/form-data; boundary=" + bd.decode())
    b = call("POST", "/batches", json.dumps({"input_file_id": f["id"], "endpoint": "/v1/chat/completions",
                                             "completion_window": "24h", "metadata": {"tag": tag}}).encode())
    json.dump({"tag": tag, "model": model, "sets": sets, "n": len(lines), "max_tokens": maxtok, "batch_id": b["id"],
               "status": b["status"]}, open(f"out/openai_batch_{tag}.json", "w"), indent=1)
    print(f"::notice::batch {tag}: {len(lines)} requests, model {model}, id {b['id']}, status {b['status']}")

elif mode == "fetch":
    tag = sys.argv[2]
    meta = json.load(open(f"state/openai_batch_{tag}.json"))
    b = call("GET", f"/batches/{meta['batch_id']}")
    meta.update({k: b.get(k) for k in ("status", "request_counts", "output_file_id", "error_file_id", "completed_at", "errors")})
    json.dump(meta, open(f"out/openai_batch_{tag}.json", "w"), indent=1)
    print(f"::notice::batch {tag}: {b['status']} {json.dumps(b.get('request_counts'))}")
    if b.get("output_file_id"):
        raw = call("GET", f"/files/{b['output_file_id']}/content", raw=True)
        enc(f"openai_raw_{tag}.jsonl.age", raw)
        usage = [json.loads(l)["response"]["body"].get("usage", {}) for l in raw.decode().splitlines() if l.strip()]
        it, ot = sum(u.get("prompt_tokens", 0) for u in usage), sum(u.get("completion_tokens", 0) for u in usage)
        print(f"::notice::{tag}: raw output saved encrypted ({len(raw)} bytes); tokens in {it} out {ot}")
    if b.get("error_file_id"):
        enc(f"openai_err_{tag}.jsonl.age", call("GET", f"/files/{b['error_file_id']}/content", raw=True))
