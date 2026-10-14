"""Ask a model to answer requests/own150.txt (150 everyday Hebrew requests written by an AI, no personal data) and save the
answers ENCRYPTED to recipient.txt (age). Nothing readable is written to this repository or to the logs: only counts,
the model name and errors. usage: python gen_answers.py gemini|groq:<model>
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

from pyrage import encrypt, x25519

prov = sys.argv[1]
reqs = [l.strip() for l in open("requests/own150.txt", encoding="utf-8") if l.strip()][:int(os.environ.get("LIMIT") or 10**6)]
recipient = x25519.Recipient.from_str(open("recipient.txt").read().strip())


UA = "hebai-public-tools/1.0 (+https://github.com/God-of-Data/hebai-public-tools)"   # Cloudflare rejects the default urllib agent


def post(url, body, headers):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": UA, **headers})
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.load(f)


def get(url, headers):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, **headers}), timeout=60) as f:
        return json.load(f)


if prov == "gemini":
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("GEMINI_API_KEY secret is empty or missing")
    H = {"x-goog-api-key": key}
    models = [m["name"] for m in get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", H)["models"]
              if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = sorted([m for m in models if "flash" in m and "lite" not in m and "image" not in m and "tts" not in m
                    and "live" not in m and "exp" not in m and "preview" not in m], reverse=True)
    rotation = [m.strip() for m in (os.environ.get("MODEL") or "").split(",") if m.strip()] or ([flash[0]] if flash else [models[0]])
    model = rotation[0]
    print("available flash models:", flash[:8], "-> using", model, flush=True)
    print(f"::notice::using {model}; flash models: {', '.join(flash[:6])}", flush=True)
    status_models = flash[:8]

    def ask(q):
        j = post(f"https://generativelanguage.googleapis.com/v1beta/{CUR['m']}:generateContent",
                 {"contents": [{"role": "user", "parts": [{"text": q}]}]}, H)
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
    CUR = {"m": model, "k": 0}


    def next_model(reason):
        """switch to the next model in the rotation; False when none is left."""
        status.setdefault("switched", []).append([CUR["m"], reason])
        CUR["k"] += 1
        if CUR["k"] >= len(rotation):
            return False
        CUR["m"] = rotation[CUR["k"]]
        print(f"::notice::switching to {CUR['m']} ({reason})", flush=True)
        return True
else:
    rotation, CUR = None, None
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        sys.exit("GROQ_API_KEY secret is empty or missing")
    model = prov.split(":", 1)[1]
    H = {"Authorization": f"Bearer {key}"}
    try:
        gm = sorted(m["id"] for m in get("https://api.groq.com/openai/v1/models", H)["data"])
        print("::notice::groq models: " + ", ".join(gm)[:900], flush=True)
        GROQ_MODELS = gm
    except Exception as e:  # noqa: BLE001
        GROQ_MODELS = [f"list failed: {e}"]

    def ask(q):
        j = post("https://api.groq.com/openai/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": q}]}, H)
        return j["choices"][0]["message"]["content"]

out, fails, status = [], 0, {"provider": prov, "errors": []}
if prov != "gemini":
    status["groq_models"] = GROQ_MODELS
os.makedirs("out", exist_ok=True)
tag = prov.replace(":", "_").replace("/", "_")
run_id = os.environ.get("GITHUB_RUN_ID", "local")
done_idx = set(json.load(open("prev_done.json"))) if os.path.exists("prev_done.json") else set()
status["done"] = sorted(done_idx)


def save_out():
    """Encrypted answers of THIS run (one file per run, so earlier runs' files are kept)."""
    if out:
        open(f"out/{tag}_{run_id}.jsonl.age", "wb").write(
            encrypt("\n".join(json.dumps(r, ensure_ascii=False) for r in out).encode(), [recipient]))


def save_status():
    """Plain status for debugging: model, counts, HTTP codes and the API's own error messages. Never any text."""
    json.dump(status, open("out/status.json", "w"), indent=1)
for i, q in enumerate(reqs):
    if i in done_idx:
        continue
    if status.get("daily_quota_hit"):
        break
    for k in range(5):
        try:
            out.append({"i": i, "model": CUR["m"] if CUR else model, "answer": ask(q)})
            status["done"].append(i)
            save_out()
            save_status()
            break
        except urllib.error.HTTPError as e:
            retry_s, quota = None, ""
            raw_body = e.read().decode(errors="replace")
            try:
                err = json.loads(raw_body).get("error", {})
                msg = err.get("message", "")[:200]
                for d in err.get("details", []):
                    if d.get("retryDelay"):
                        retry_s = float(str(d["retryDelay"]).rstrip("s") or 0)
                    for v in d.get("violations", []):
                        quota = v.get("quotaId", "") or quota
            except Exception:  # noqa: BLE001
                msg = "non-JSON body: " + raw_body[:150]
            msg = f"{msg} [quota {quota}]"
            if e.code == 429 and "PerDay" in quota:
                if CUR and next_model("daily quota"):
                    continue
                status["daily_quota_hit"] = quota
                save_status()
                break
            if e.code == 503 and k >= 2 and CUR and next_model("overloaded"):
                continue
            status["errors"].append({"item": i, "code": e.code, "message": msg})
            print(f"::warning::item {i} HTTP {e.code}: {msg}", flush=True)   # annotation: readable through the API
            save_status()
            if e.code in (400, 401, 402, 403, 404):
                status["fatal"] = True
                save_status()
                sys.exit(f"HTTP {e.code}: {msg}")
            wait = (retry_s + 2) if retry_s else (20 * (k + 1) if e.code in (429, 500, 503) else 5)
            print(f"item {i}: HTTP {e.code}, retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::item {i}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(5)
    else:
        fails += 1
    time.sleep(float(os.environ.get("PAUSE", "4")))
    if i % 25 == 0:
        print(f"{i + 1}/{len(reqs)} done, {fails} failed", flush=True)
status.update({"model": model, "saved_this_run": len(out), "failed": fails, "total_done": len(status["done"])})
save_status()
save_out()
print(f"::notice::saved {len(out)} answers (encrypted), {fails} failed, model {model}")
