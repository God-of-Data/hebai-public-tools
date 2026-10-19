"""#96 step 1: herzl-like human posts from HPLT 2.0 cleaned heb_Hebr. Files read in a fixed shuffled order starting at
<start> (the #93-N pool used the first 12). Crawl < 2022-11-01, <= 3 docs per host, 60-300 words, >= 4 "!" and >= 2 hits
of a fixed emotional-word list (a SAMPLING filter, never a model feature). Full text AGE-ENCRYPTED to recipient.txt.
usage: python hplt_emo.py <start> <n_files> <max_docs>"""
import collections, json, os, random, re, sys, urllib.parse
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from pyrage import encrypt, x25519
REPO = "HPLT/HPLT2.0_cleaned"; S, NF, MAXD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
PAT = re.compile("מזל טוב|יום הולדת|חג שמח|שנה טובה|זיכרון|לזכר|מרגש|ברכות|אוהבת|אוהב|תודה")
rcp = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("heb_Hebr/"))
random.Random(93).shuffle(files); per_host = collections.Counter(); out = []; seen = 0
print(f"::notice::{len(files)} files; reading {S}..{S + NF}")
for f in files[S:S + NF]:
    t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=["id", "u", "ts", "collection", "text"]).to_pylist()
    for r in t:
        seen += 1; x = r["text"]; n = len(x.split())
        if not r["ts"] or str(r["ts"]) >= "2022-11-01" or n < 60 or n > 330: continue
        if x.count("!") < 4 or len(PAT.findall(x)) < 2: continue
        h = urllib.parse.urlparse(r["u"]).netloc.lower()
        if per_host[h] >= 3: continue
        per_host[h] += 1; out.append({"id": "hpltemo_" + str(r["id"])[:24], "url": r["u"], "unit": h, "ts": str(r["ts"]), "collection": r["collection"], "text": x})
        if len(out) >= MAXD: break
    print(f"::notice::{f}: kept {len(out)} of {seen}", flush=True)
    if len(out) >= MAXD: break
    try: os.remove(hf_hub_download(REPO, f, repo_type="dataset"))
    except Exception: pass
os.makedirs("out", exist_ok=True)
open("out/hplt_emo.jsonl.age", "wb").write(encrypt("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out).encode(), [rcp]))
print(f"::notice::kept {len(out)} docs from {len(per_host)} hosts of {seen} read")
