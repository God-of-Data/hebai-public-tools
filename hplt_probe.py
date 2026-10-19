"""#93-N(b) probe (no content leaves unencrypted): HPLT 2.0 cleaned, heb_Hebr. Lists the files, downloads ONE parquet,
prints its schema, row count, date range of any timestamp column and host counts. Nothing readable is written."""
import collections, json, sys, urllib.parse
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
REPO = "HPLT/HPLT2.0_cleaned"
api = HfApi(); files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("heb_Hebr/")]
print("::notice::heb_Hebr files", len(files), files[:3])
p = hf_hub_download(REPO, files[0], repo_type="dataset")
t = pq.read_table(p); print("::notice::schema", t.schema.names, "rows", t.num_rows)
cols = t.schema.names
for c in cols:
    if c in ("text",): continue
    v = t.column(c).to_pylist()[:3]; print("::notice::col", c, "examples", json.dumps([str(x)[:80] for x in v]))
if "url" in cols:
    hosts = collections.Counter(urllib.parse.urlparse(u).netloc for u in t.column("url").to_pylist())
    print("::notice::hosts", len(hosts), "top", hosts.most_common(5))
for c in cols:
    if c.lower() in ("ts", "timestamp", "date", "crawl_date", "crawl_id"):
        v = sorted(str(x) for x in t.column(c).to_pylist() if x is not None); print("::notice::range", c, v[0], v[-1])
