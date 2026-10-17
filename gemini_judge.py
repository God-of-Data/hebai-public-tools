"""Free-tier Gemini as a judge / teacher (#81). Input: requests/<set>.jsonl.age — one line per request {"texts": [...]}
(public texts only), decrypted in memory with REQ_AGE_KEY. Each request asks for one AI-probability (0-100) per text
as a JSON list. One model for the whole set: the first in a fixed order that answers (Fable #81 amendment). Raw replies
are encrypted to recipient.txt (out/raw_<set>_<n>.jsonl.age); out/status.json keeps the done request numbers and the
model per request (no content). Resumes where the previous run stopped. usage: python gemini_judge.py <set>"""
import json, os, re, sys, time, traceback, urllib.error, urllib.request
from pyrage import decrypt, encrypt, x25519
os.makedirs("out", exist_ok=True)
def _hook(t, v, tb):   # the run log is not readable from here: report the failure as an annotation (no content)
    print("::error::" + " | ".join(traceback.format_exception(t, v, tb)[-3:]).replace("\n", " ")[:900], flush=True)
sys.excepthook = _hook

SET = sys.argv[1]
KEY = os.environ["GEMINI_API_KEY"].strip(); H = {"x-goog-api-key": KEY, "Content-Type": "application/json"}
ORDER = [m.strip() for m in os.environ.get("MODELS", "models/gemini-3.8-flash,models/gemini-3.7-flash,models/gemini-3.6-flash,"
                                            "models/gemini-3.5-flash,models/gemini-flash-latest").split(",") if m.strip()]
RCP = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
reqs = [json.loads(l) for l in decrypt(open(f"requests/{SET}.jsonl.age", "rb").read(),
                                       [x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())]).decode().splitlines() if l.strip()]
os.makedirs("out", exist_ok=True)
st = json.load(open("out/status.json")) if os.path.exists("out/status.json") else {"done": [], "model": {}, "errors": []}
PROMPT = ("Below are {n} short Hebrew messages, numbered [1] to [{n}]. For EACH message independently, estimate the "
          "probability (0-100) that it was written by an AI chatbot rather than by a person. Judge each message on its own; "
          "do not compare them with each other. Reply with a JSON list of {n} integers, in order, and nothing else.\n\n")


def ask(model, q):
    body = {"contents": [{"role": "user", "parts": [{"text": q}]}]}
    r = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent",
                               data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(r, timeout=180) as f:
        j = json.load(f)
    return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]), j.get("usageMetadata", {})


# Fable #81 amendment: ONE model for the whole set. The first model in ORDER that answers is fixed (status["fixed"]);
# later runs use it; on its daily quota (429) the run stops and resumes after the reset; overload (503) waits and retries.
res, t0 = [], time.time()
cands = [st["fixed"]] if st.get("fixed") else ORDER
stop = False
for n, rq in enumerate(reqs):
    if n in st["done"] or stop:
        continue
    q = PROMPT.format(n=len(rq["texts"])) + "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(rq["texts"]))
    attempt, waits = 0, 0
    while attempt < 2 and not stop:
        model = st.get("fixed") or cands[0]
        try:
            txt, use = ask(model, q)
        except urllib.error.HTTPError as e:
            code = e.code; st["errors"].append([n, model, code, e.read()[:120].decode(errors="replace")])
            if code == 429 and st.get("fixed"):
                print(f"::notice::{model}: daily quota reached; resume after the reset", flush=True); stop = True; break
            if not st.get("fixed") and code in (429, 404, 403):
                cands = cands[1:]
                if not cands: print("::error::no model answered", flush=True); stop = True
                continue
            waits += 1
            if waits > 6: print(f"::notice::{model}: still {code}; resume later", flush=True); stop = True; break
            time.sleep(30 * waits); continue
        except Exception as e:  # noqa: BLE001
            st["errors"].append([n, model, "exc", repr(e)[:120]]); waits += 1
            if waits > 6: stop = True; break
            time.sleep(20); continue
        st.setdefault("fixed", model)
        m = re.search(r"\[[\d\s,.]*\]", txt or "")
        try: vals = json.loads(m.group(0)) if m else None
        except ValueError: vals = None
        good = isinstance(vals, list) and len(vals) == len(rq["texts"])
        res.append({"n": n, "model": model, "attempt": attempt, "text": txt, "usage": use, "parsed": good})
        if good or attempt == 1:
            st["done"].append(n); st["model"][str(n)] = model; break
        attempt += 1
    time.sleep(4)
part = len([f for f in os.listdir("out") if f.startswith(f"raw_{SET}_")])
if res:
    open(f"out/raw_{SET}_{part}.jsonl.age", "wb").write(encrypt("\n".join(json.dumps(r, ensure_ascii=False) for r in res).encode(), [RCP]))
json.dump(st, open("out/status.json", "w"))
print(f"::notice::{SET}: {len(st['done'])}/{len(reqs)} requests done, {sum(r['parsed'] for r in res)} parsed this run, "
      f"models {sorted(set(st['model'].values()))}, {(time.time() - t0) / 60:.1f} min")
