"""One request, no retries: tells whether the key works and, if not, why. Writes out/status.json (model names, HTTP
code, the API's own error message, answer length) — never any request or answer text."""
import json
import os
import sys
import urllib.error
import urllib.request

st = {}
os.makedirs("out", exist_ok=True)


def done(code=0):
    json.dump(st, open("out/status.json", "w"), indent=1)
    print("::notice::" + json.dumps(st)[:900], flush=True)
    sys.exit(code)


key = os.environ.get("GEMINI_API_KEY", "")
st["key_present"] = bool(key)
if not key:
    done(1)
H = {"x-goog-api-key": key, "Content-Type": "application/json"}
try:
    with urllib.request.urlopen(urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers=H), timeout=30) as f:
        models = [m["name"] for m in json.load(f)["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]
    st["models"] = [m for m in models if "flash" in m or "pro" in m]
except urllib.error.HTTPError as e:
    st["list_models_error"] = [e.code, e.read().decode()[:400]]
    done(1)
LIST = [m.strip() for m in os.environ.get("MODEL", "").split(",") if m.strip()]
model = (LIST[0] if LIST else "") or next((m for m in sorted(st["models"], reverse=True)
                                         if "lite" not in m and "preview" not in m and "exp" not in m and "image" not in m
                                         and "tts" not in m and "live" not in m), st["models"][0] if st["models"] else "")
st["model"] = model
REQS = 1 if LIST else int(os.environ.get("PROBE_N", "3"))   # a model list = 1 request each, to spare the daily quota
body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "כתוב משפט אחד על מזג האוויר."}]}]}).encode()
st["calls"] = []
cands = LIST or [m for m in st["models"] if not any(x in m for x in ("image", "tts", "live", "omni"))]
for m in cands:
    for k in range(REQS):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent",
                                                               data=body, headers=H), timeout=60) as f:
                j = json.load(f)
            st["calls"].append([m, k, "ok", len("".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]))])
        except urllib.error.HTTPError as e:
            err = json.loads(e.read().decode() or "{}").get("error", {})
            viol = [v for d in err.get("details", []) for v in d.get("violations", [])]
            retry = [d.get("retryDelay") for d in err.get("details", []) if d.get("retryDelay")]
            st["calls"].append([m, k, e.code, [(v.get("quotaMetric", "").split("/")[-1], v.get("quotaId"), v.get("quotaValue")) for v in viol], retry,
                                err.get("message", "")[:160]])
st["ok"] = any(c[2] == "ok" for c in st["calls"])
done(0)
