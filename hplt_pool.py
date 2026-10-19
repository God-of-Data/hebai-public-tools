"""#93-N(b) fresh human pool from HPLT 2.0 cleaned heb_Hebr: random files (fixed seed), documents crawled before
2022-11-01, at most 3 per host, Hebrew documents of >= 60 words. Full text kept, AGE-ENCRYPTED to recipient.txt; nothing
readable is written. usage: python hplt_pool.py <n_files> <max_docs>"""
import collections, json, os, random, sys, urllib.parse
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from pyrage import encrypt, x25519
REPO = "HPLT/HPLT2.0_cleaned"; NF, MAXD = int(sys.argv[1]), int(sys.argv[2])
rcp = x25519.Recipient.from_str([w for w in open("recipient.txt").read().split() if w.startswith("age1")][0])
files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("heb_Hebr/"))
random.Random(93).shuffle(files); per_host = collections.Counter(); out = []; seen = 0
for f in files[:NF]:
    t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=["id", "u", "ts", "collection", "text"]).to_pylist()
    random.Random(f).shuffle(t)
    for r in t:
        seen += 1
        if not r["ts"] or str(r["ts"]) >= "2022-11-01" or len(r["text"].split()) < 60: continue
        h = urllib.parse.urlparse(r["u"]).netloc.lower()
        if per_host[h] >= 3: continue
        per_host[h] += 1; out.append({"id": "hplt_" + str(r["id"])[:24], "url": r["u"], "unit": h, "ts": str(r["ts"]), "collection": r["collection"], "text": r["text"]})
        if len(out) >= MAXD: break
    print(f"::notice::{f}: kept {len(out)} of {seen}", flush=True)
    if len(out) >= MAXD: break
os.makedirs("out", exist_ok=True)
open("out/hplt_pool.jsonl.age", "wb").write(encrypt("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out).encode(), [rcp]))
print(f"::notice::kept {len(out)} docs from {len(per_host)} hosts; collections {dict(collections.Counter(o['collection'] for o in out))}")
