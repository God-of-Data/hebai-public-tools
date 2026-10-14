"""GitHub Models probe: lists the catalog and sends ONE short request per chat model, using the workflow's own
GITHUB_TOKEN (no key, no card). Writes out/status.json: model id, publisher, HTTP code, the API's error message,
rate-limit headers and answer LENGTH only — never any answer text."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

UA = "hebai-public-tools/1.0"
TOK = os.environ.get("GITHUB_TOKEN", "")
H = {"Authorization": f"Bearer {TOK}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
     "Content-Type": "application/json", "User-Agent": UA}
os.makedirs("out", exist_ok=True)
st = {"token_present": bool(TOK), "calls": []}


def save():
    json.dump(st, open("out/status.json", "w"), indent=1)


try:
    with urllib.request.urlopen(urllib.request.Request("https://models.github.ai/catalog/models", headers=H), timeout=30) as f:
        cat = json.load(f)
except urllib.error.HTTPError as e:
    st["catalog_error"] = [e.code, e.read().decode(errors="replace")[:300]]
    save()
    sys.exit(1)
st["catalog"] = [{k: m.get(k) for k in ("id", "publisher", "rate_limit_tier", "supported_output_modalities")} for m in cat]
chat = [m for m in cat if "text" in (m.get("supported_output_modalities") or []) and "embed" not in m["id"]]
body = json.dumps({"messages": [{"role": "user", "content": "כתוב משפט אחד על מזג האוויר."}], "max_tokens": 200})
for m in chat:
    payload = json.loads(body)
    payload["model"] = m["id"]
    req = urllib.request.Request("https://models.github.ai/inference/chat/completions", data=json.dumps(payload).encode(), headers=H)
    try:
        with urllib.request.urlopen(req, timeout=90) as f:
            j = json.load(f)
            hdr = {k: v for k, v in f.headers.items() if "ratelimit" in k.lower()}
        txt = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        st["calls"].append([m["id"], m.get("rate_limit_tier"), "ok", len(txt), hdr])
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            msg = json.loads(raw).get("error", {}).get("message", "")[:200]
        except Exception:  # noqa: BLE001
            msg = raw[:200]
        hdr = {k: v for k, v in e.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
        st["calls"].append([m["id"], m.get("rate_limit_tier"), e.code, msg, hdr])
    except Exception as e:  # noqa: BLE001
        st["calls"].append([m["id"], m.get("rate_limit_tier"), type(e).__name__, str(e)[:150], {}])
    save()
    time.sleep(4)
st["ok"] = [c[0] for c in st["calls"] if c[2] == "ok"]
save()
print("::notice::ok models: " + ", ".join(st["ok"])[:900])
