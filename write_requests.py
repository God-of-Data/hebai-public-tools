"""Ask an open model (Groq) to write everyday Hebrew requests to a chatbot, so the training requests are not all phrased
by Claude. Requests only (no answers); they are generic and contain no personal data. Writes out/requests_<model>.txt."""
import json
import os
import re
import sys
import time
import urllib.request

UA = "hebai-public-tools/1.0 (+https://github.com/God-of-Data/hebai-public-tools)"
key = os.environ["GROQ_API_KEY"]
model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
KINDS = [
    "מכתבים והודעות רשמיות (לעירייה, לוועד בית, לבנק, לבית ספר, מכתב תלונה, הודעה לדיירים או ללקוחות)",
    "הודעות וואטסאפ קצרות לכתוב למישהו (לבוס, לשכן, לקבוצת הורים, לחבר, לבעל דירה, למוכר)",
    "בקשות להסביר משהו (כסף, זכויות, מדע, טכנולוגיה, בריאות, מושגים) לפעמים לילד או בפשטות",
    "בקשות לעצה או התלבטות אישית (זוגיות, הורות, עבודה, כסף, חברים, בריאות)",
]
out = []
for kind in KINDS:
    for half in ("רגיל", "בלי עיצוב"):
        extra = ("בערך בחצי מהבקשות המשתמש מבקש במפורש תשובה בטקסט רגיל בלי כוכביות, בלי רשימות ובלי כותרות."
                 if half == "בלי עיצוב" else "חלק מהבקשות קצרות מאוד וחלק עם רקע אישי.")
        prompt = (f"כתוב 20 בקשות שונות בעברית שאנשים רגילים בישראל מקלידים לצ'טבוט, מהסוג: {kind}. "
                  f"כתוב כמו שאנשים באמת כותבים בטלפון: לפעמים סלנג, שגיאות קטנות, לפעמים רשמי. {extra} "
                  "בלי מספור, בלי מרכאות, בקשה אחת בכל שורה, ושום דבר מלבד הבקשות.")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 3000}
        for k in range(5):
            try:
                req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "User-Agent": UA,
                                                      "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=120) as f:
                    text = json.load(f)["choices"][0]["message"]["content"]
                break
            except Exception as e:  # noqa: BLE001
                print(f"retry {k}: {str(e)[:120]}", flush=True)
                time.sleep(20 * (k + 1))
        else:
            continue
        lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip().strip('"') for l in text.splitlines()]
        out += [l for l in lines if len(l.split()) >= 3 and re.search(r"[א-ת]", l)]
        time.sleep(5)
seen, uniq = set(), []
for l in out:
    if l not in seen:
        seen.add(l)
        uniq.append(l)
os.makedirs("out", exist_ok=True)
open(f"out/requests_{model.replace('/', '_')}.txt", "w", encoding="utf-8").write("\n".join(uniq) + "\n")
print(f"::notice::wrote {len(uniq)} requests with {model}")
