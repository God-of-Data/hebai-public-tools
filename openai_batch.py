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

if mode == "write":   # Fable #28: half of the new request sets written by ChatGPT (requests only, generic, no personal data)
    NUM = ("כתוב 20 בקשות שונות בעברית שאנשים בישראל מקלידים לצ'אטבוט כדי לקבל הודעת וואטסאפ מסודרת עם פירוט מספרי, "
           "בנושא: {k}. כל בקשה מבקשת שהתשובה תהיה הודעת וואטסאפ עם שורות קצרות ומספרים. כתוב כמו שאנשים כותבים בטלפון. "
           "אסור לכלול: מסעדות, המבורגר, ארוחות, עלות מנה או אוכל. בלי מספור, בקשה אחת בכל שורה, ושום דבר מלבד הבקשות.")
    EXP = ("כתוב 20 בקשות שונות בעברית שאנשים בישראל מקלידים לצ'אטבוט כדי שיכתוב להם פוסט אישי לרשת חברתית אחרי אירוע "
           "מהסוג: {k}, כפי שהם היו מפרסמים, בשורות קצרות. אל תזכיר אימוג'ים או סימנים. אסור לכלול: יום הזיכרון, טקסי זיכרון, "
           "הר הרצל, סיורי לילה עם תלמידים. בלי מספור, בקשה אחת בכל שורה, ושום דבר מלבד הבקשות.")
    kinds = [("numeric", NUM, k) for k in ("תקציבים וחלוקת הוצאות בין אנשים", "מחירונים והצעות מחיר של עסקים קטנים",
                                          "השוואות מחירים של שירותים ומוצרים", "לוחות זמנים ותורנויות עם שעות ותאריכים")]
    kinds += [("expressive", EXP, k) for k in ("הישג אישי או ספורטיבי", "אירוע משפחתי משמח", "שינוי גדול בחיים כמו מעבר, עבודה חדשה או פרידה")]
    out = []
    for kind, tmpl, k in kinds:
        r = call("POST", "/chat/completions", json.dumps({"model": sys.argv[3], "messages": [{"role": "user", "content": tmpl.format(k=k)}],
                                                          "max_completion_tokens": 4000}).encode())
        out.append({"kind": kind, "topic": k, "response": r})
        print(f"::notice::{kind} / {k}: {len((r['choices'][0]['message'].get('content') or '').splitlines())} lines")
    json.dump(out, open(f"out/requests_written_{sys.argv[2]}.json", "w"), ensure_ascii=False, indent=1)

if mode == "cancel":   # OpenAI keeps the completed results of a cancelled batch (and bills only those)
    tag = sys.argv[2]
    meta = json.load(open(f"state/openai_batch_{tag}.json"))
    b = call("POST", f"/batches/{meta['batch_id']}/cancel", b"{}")
    meta["status"] = b["status"]; meta["cancel_requested"] = True
    json.dump(meta, open(f"out/openai_batch_{tag}.json", "w"), indent=1)
    print(f"::notice::batch {tag}: cancel requested, status {b['status']} {json.dumps(b.get('request_counts'))}")

elif mode == "resubmit":   # new batch <newtag> with only the requests of <tag> that have no completed answer
    newtag, tag = sys.argv[2], sys.argv[3]   # workflow: tag = new tag, model field = source tag
    meta = json.load(open(f"state/openai_batch_{tag}.json"))
    b = call("GET", f"/batches/{meta['batch_id']}")
    done = set()
    if b.get("output_file_id"):   # read in memory only (custom ids); the raw file itself is saved by fetch
        for l in call("GET", f"/files/{b['output_file_id']}/content", raw=True).decode().splitlines():
            if l.strip():
                r = json.loads(l)
                if (r.get("response") or {}).get("status_code") == 200: done.add(r["custom_id"])
    sys.argv = [sys.argv[0], "submit", newtag, meta["model"], ",".join(meta["sets"]), str(meta.get("max_tokens", 1200))]
    SKIP = {c.split("|", 1)[1] for c in done}
    print(f"::notice::{tag}: {len(done)} completed; resubmitting the rest as {newtag}")
    mode = "submit"

if mode == "submit":
    tag, model, sets = sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    maxtok = int(sys.argv[5]) if len(sys.argv) > 5 else 1200
    key = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())
    lines = []
    for s in sets:
        p = next(x for x in (f"requests/{s}.jsonl.age", f"requests/{s}.txt.age", f"requests/{s}.txt") if os.path.exists(x))
        blob = open(p, "rb").read()
        txt = decrypt(blob, [key]).decode() if p.endswith(".age") else blob.decode()
        qs = ([json.loads(l)["prompt"] for l in txt.split("\n") if l.strip()] if ".jsonl" in p   # multi-line prompts (Fable #52)
              else [l.strip() for l in txt.split("\n") if l.strip()])
        for i, q in enumerate(qs):
            if f"{s}|{i}" in globals().get("SKIP", set()): continue
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

if mode == "fetch":
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
