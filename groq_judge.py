"""Free-tier Groq as a judge / teacher (#81, owner 2026-10-17). Same request sets as gemini_judge.py (one line per request
{"texts": [...]}, public texts only, decrypted in memory with REQ_AGE_KEY). ONE fixed model (MODEL, default
openai/gpt-oss-120b, reasoning_effort low). Rate limits (429) wait and retry; after too many the run stops and resumes
later. Raw replies encrypted to recipient.txt (out/raw_<set>_<n>.jsonl.age); out/status.json keeps done request numbers
(no content). usage: python groq_judge.py <set>"""
import json, os, re, sys, time, traceback, urllib.error, urllib.request
from pyrage import decrypt, encrypt, x25519
os.makedirs("out", exist_ok=True)
def _hook(t, v, tb):
    print("::error::" + " | ".join(traceback.format_exception(t, v, tb)[-3:]).replace("\n", " ")[:900], flush=True)
sys.excepthook = _hook
SET = sys.argv[1]; MODEL = os.environ.get("MODEL") or "openai/gpt-oss-120b"
H = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}", "Content-Type": "application/json", "User-Agent": "hebai-judge/1.0"}
RCP = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
reqs = [json.loads(l) for l in decrypt(open(f"requests/{SET}.jsonl.age", "rb").read(),
                                       [x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())]).decode().splitlines() if l.strip()]
st = json.load(open("out/status.json")) if os.path.exists("out/status.json") else {"done": [], "model": {}, "errors": []}
PROMPT = ("Below are {n} short Hebrew messages, numbered [1] to [{n}]. For EACH message independently, estimate the "
          "probability (0-100) that it was written by an AI chatbot rather than by a person. Judge each message on its own; "
          "do not compare them with each other. Reply with a JSON list of {n} integers, in order, and nothing else.\n\n")
def ask(q):
    body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_completion_tokens": 3000, "reasoning_effort": os.environ.get("REASONING") or "low"}
    r = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(r, timeout=180) as f:
        j = json.load(f)
    return j["choices"][0]["message"]["content"], j.get("usage", {})
res, t0, stop = [], time.time(), False
for n, rq in enumerate(reqs):
    if n in st["done"] or stop:
        continue
    q = PROMPT.format(n=len(rq["texts"])) + "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(rq["texts"]))
    attempt, waits = 0, 0
    while attempt < 2 and not stop:
        try:
            txt, use = ask(q)
        except urllib.error.HTTPError as e:
            st["errors"].append([n, MODEL, e.code, e.read()[:160].decode(errors="replace")]); waits += 1
            if waits > 15: print(f"::notice::{MODEL}: still {e.code}; resume later", flush=True); stop = True; break
            time.sleep(min(120, 20 * waits)); continue
        except Exception as e:  # noqa: BLE001
            st["errors"].append([n, MODEL, "exc", repr(e)[:160]]); waits += 1
            if waits > 15: stop = True; break
            time.sleep(20); continue
        m = re.search(r"\[[\d\s,.]*\]", txt or "")
        try: vals = json.loads(m.group(0)) if m else None
        except ValueError: vals = None
        good = isinstance(vals, list) and len(vals) == len(rq["texts"])
        res.append({"n": n, "model": MODEL, "attempt": attempt, "text": txt, "usage": use, "parsed": good})
        if good or attempt == 1:
            st["done"].append(n); st["model"][str(n)] = MODEL; break
        attempt += 1
    time.sleep(float(os.environ.get("PAUSE") or 8))
st["fixed"] = MODEL
part = len([f for f in os.listdir("out") if f.startswith(f"raw_{SET}_")])
if res:
    open(f"out/raw_{SET}_{part}.jsonl.age", "wb").write(encrypt("\n".join(json.dumps(r, ensure_ascii=False) for r in res).encode(), [RCP]))
json.dump(st, open("out/status.json", "w"))
print(f"::notice::{SET}: {len(st['done'])}/{len(reqs)} requests done, {sum(r['parsed'] for r in res)} parsed this run, model {MODEL}, "
      f"errors {len(st['errors'])}, {(time.time() - t0) / 60:.1f} min")
