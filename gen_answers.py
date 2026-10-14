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


def post(url, body, headers):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.load(f)


def get(url, headers):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as f:
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
    model = os.environ.get("MODEL") or (flash[0] if flash else models[0])
    print("available flash models:", flash[:8], "-> using", model, flush=True)
    print(f"::notice::using {model}; flash models: {', '.join(flash[:6])}", flush=True)
    status_models = flash[:8]

    def ask(q):
        j = post(f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent",
                 {"contents": [{"role": "user", "parts": [{"text": q}]}]}, H)
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
else:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        sys.exit("GROQ_API_KEY secret is empty or missing")
    model = prov.split(":", 1)[1]
    H = {"Authorization": f"Bearer {key}"}

    def ask(q):
        j = post("https://api.groq.com/openai/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": q}]}, H)
        return j["choices"][0]["message"]["content"]

out, fails, status = [], 0, {"provider": prov, "errors": []}
os.makedirs("out", exist_ok=True)


def save_status():
    """Plain status for debugging: model, counts, HTTP codes and the API's own error messages. Never any text."""
    json.dump(status, open("out/status.json", "w"), indent=1)
for i, q in enumerate(reqs):
    for k in range(6):
        try:
            out.append({"i": i, "model": model, "answer": ask(q)})
            break
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("error", {}).get("message", "")[:300]
            except Exception:  # noqa: BLE001
                msg = ""
            status["errors"].append({"item": i, "code": e.code, "message": msg})
            print(f"::warning::item {i} HTTP {e.code}: {msg}", flush=True)   # annotation: readable through the API
            save_status()
            if e.code in (400, 401, 403, 404):
                status["fatal"] = True
                save_status()
                sys.exit(f"HTTP {e.code}: {msg}")
            wait = 10 * (k + 1) if e.code == 429 else 5
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
status.update({"model": model, "saved": len(out), "failed": fails})
save_status()
tag = prov.replace(":", "_").replace("/", "_")
open(f"out/{tag}.jsonl.age", "wb").write(encrypt("\n".join(json.dumps(r, ensure_ascii=False) for r in out).encode(), [recipient]))
print(f"::notice::saved {len(out)} answers (encrypted), {fails} failed, model {model}")
