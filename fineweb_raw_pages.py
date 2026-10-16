"""RAW PAGES re-run of the human FineWeb-2 pool (2026-10-15). Same selection as the original pool (same row groups,
filters, host caps and passage cut, so the same documents are kept), but it ALSO saves the full raw page text of every
document before the cutoff that is not excluded, so later cuts (e.g. keeping expressive short lines) can be made
locally without downloading again. All output is age-encrypted to recipient.txt; nothing readable is written.

Original docstring:
Bulk human pool from FineWeb-2 heb_Hebr (runs on GitHub Actions; needs internet).

Scans the parquet files given by --part/--parts (file i goes to part i % parts), keeps documents crawled before
2022-11-30, cuts one passage of 40–300 words per document (whole paragraphs, start chosen by the doc id), and
applies the same filters that every human text gets (decided 2026-10-13, D4):
  - Hebrew letters >= 75% of letters
  - sentences with phone numbers, e-mails or 9-digit IDs are dropped (never replaced by a placeholder token)
  - boilerplate lines (copyright, cookies, share/comment buttons) dropped; menu-like passages (mostly short lines) dropped
  - keyword-stuffed pages dropped (one content word > 8% of words)
  - sensitive sexual content dropped (same regex as the forum sources)
  - per-host cap within the part (--host-cap)
Writes data/pool/fineweb_<part>.jsonl.gz: {id, url, host, genre, date, dump, n_words, text}.

usage: python scripts/pool_fineweb.py --part 0 --parts 8 --max-files 4 --host-cap 40
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hebai.genre import exclude_reason, genre_v2, host_of  # noqa: E402
from hebai.splits import is_test_host  # noqa: E402

CUTOFF = "2022-11-30"
HEB = re.compile(r"[א-ת]")
LETTER = re.compile(r"[^\W\d_]")
PHONE = re.compile(r"(?<!\d)(?:\+972[-\s]?|0)(?:5\d|[2-489]|7\d)[-\s]?\d{3}[-\s]?\d{3,4}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
IDNUM = re.compile(r"(?<!\d)\d{9}(?!\d)")
SENT = re.compile(r"(?<=[.!?])\s+")
BOILER = re.compile(r"כל הזכויות שמורות|זכויות יוצרים|עוגיות|cookies|תנאי שימוש|מדיניות פרטיות|הוסף תגובה|הוספת תגובה|"
                    r"שתף ב|שתפו|לחצו כאן|לחץ כאן|הירשמו|הרשמו לניוזלטר|צור קשר|קרא עוד|המשך לקרוא|תגובות\s*\(\d+\)|"
                    r"©|Powered by|Copyright", re.I)
SENSITIVE = re.compile("סקס|יחסי מין|יחסים מיני|אוננ|אורגז|פורנו|זיון|לזיין|לשכב איתו|לשכב איתה|שכבנו|שכבתי עם|בתולים|"
                       "בתולה|עירום|ערום|מיני(?:ת|ים|ות)?\\b|אקט|נגיעות|מזמוז|התמזמז|ציצי|זין\\b|כוס\\b|קונדום|גלולת|"
                       "ליווי|נערות ליווי|דיסקרטי|הימורים|קזינו")
STOP = set("של את על עם זה זו לא כי גם או אם הוא היא הם הן אני אנחנו אתה את יש אין כל מה מי לכל אבל רק עוד כמו "
           "אחד אחת היה היו להיות שלא שלו שלה שלי בין לפי אשר כך כן עד אל".split())


def raw_passage(text: str, lo=40, hi=300):
    """Test-only cut (Fable review #5, B1): the page from its first line, structure kept (short lines, list markers,
    colons); only sentences with personal data are dropped. 40–300 words, whole lines."""
    out, n = [], 0
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
        sents = [x for x in SENT.split(l) if x and not (PHONE.search(x) or EMAIL.search(x) or IDNUM.search(x))]
        if not sents:
            continue
        l = " ".join(sents)
        w = len(l.split())
        if out and n + w > hi:
            break
        out.append(l)
        n += w
    return "\n".join(out) if lo <= n <= hi + 40 else None


def clean_passage(text: str, seed: int, lo=40, hi=300):
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l and not BOILER.search(l)]
    if not lines:
        return None
    short = sum(1 for l in lines if len(l.split()) < 5)
    if len(lines) >= 4 and short / len(lines) > 0.5:
        return None  # menu / link list
    paras = []
    for l in lines:
        sents = [s for s in SENT.split(l) if s and not (PHONE.search(s) or EMAIL.search(s) or IDNUM.search(s))]
        if sents and len(" ".join(sents).split()) >= 4:  # titles, menu items, buttons are not passage text
            paras.append(" ".join(sents))
    if not paras:
        return None
    start = seed % len(paras)
    # walk back to a start that leaves room for >= lo words
    while start > 0 and sum(len(p.split()) for p in paras[start:]) < lo:
        start -= 1
    out, n = [], 0
    for p in paras[start:]:
        w = len(p.split())
        if n + w > hi:
            if n >= lo:
                break
            # cut a long paragraph at a sentence boundary
            acc = []
            for s in SENT.split(p):
                if n + len(s.split()) > hi:
                    break
                acc.append(s)
                n += len(s.split())
            if acc:
                out.append(" ".join(acc))
            break
        out.append(p)
        n += w
    if n < lo:
        return None
    return "\n".join(out)


def ok_text(t: str) -> bool:
    letters = LETTER.findall(t)
    if not letters or sum(1 for c in letters if HEB.match(c)) / len(letters) < 0.75:
        return False
    if SENSITIVE.search(t):
        return False
    words = [w for w in re.findall(r"[א-ת]{3,}", t) if w not in STOP]
    if words:
        top = collections.Counter(words).most_common(1)[0][1]
        if len(words) >= 40 and top / len(words) > 0.08:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=6)
    ap.add_argument("--rg-per-file", type=int, default=12)
    ap.add_argument("--host-cap", type=int, default=40)
    ap.add_argument("--max-keep", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--fresh-round", type=int, default=4, help="5 (Fable #28): also exclude the fresh #4 row groups; 7 (Fable #46): also fresh #6")
    ap.add_argument("--fresh", action="store_true", help="fresh #4 (2026-10-16): NEW row groups only (none the original "
                    "pool sampled), test hosts only, raw pages saved; output tag fresh")
    ap.add_argument("--raw-test", action="store_true", help="test hosts only, raw_passage cut (structure kept)")
    a = ap.parse_args()
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    t0 = time.time()
    files = sorted(f for f in HfApi().list_repo_files("HuggingFaceFW/fineweb-2", repo_type="dataset")
                   if f.startswith("data/heb_Hebr/train/") and f.endswith(".parquet"))
    # FineWeb-2 heb_Hebr has 6 big files (~2.9M docs, ~2,900 row groups each); every part takes its own random
    # row groups from every file, so each part spreads over all dumps and hosts
    mine = files
    os.makedirs("data/pool", exist_ok=True)
    tag = "fineweb_raw" if a.raw_test else "fineweb"
    import io
    buf = io.StringIO()            # kept passages (same as the original pool)
    raw = io.StringIO()            # every candidate page, full text
    out = buf
    hosts, stats, seen = collections.Counter(), collections.Counter(), set()
    kept = 0
    for fname in mine:
        if kept >= a.max_keep or (time.time() - t0) / 60 > a.max_minutes:
            break
        # remote read, row group by row group (downloading whole shards killed probe runners)
        fh = HfFileSystem().open(f"datasets/HuggingFaceFW/fineweb-2/{fname}", "rb", block_size=8 * 2**20)
        pf = pq.ParquetFile(fh)
        cols = [c for c in ("id", "url", "date", "dump", "text") if c in pf.schema_arrow.names]
        import random
        rnd = random.Random(1000 * a.part + files.index(fname))
        rgs = [g for g in range(pf.num_row_groups) if g % a.parts == a.part]
        if a.fresh:   # the original pool's row groups for every original part (8 parts, 15 per file) are excluded
            used = set()
            for op in range(8):
                r0 = random.Random(1000 * op + files.index(fname))
                cand = [g for g in range(pf.num_row_groups) if g % 8 == op]
                used.update(r0.sample(cand, min(15, len(cand))))
            if a.fresh_round >= 5:   # fresh #4 used 4 parts x 6 row groups per file with seed 777000
                for op in range(4):
                    r4 = random.Random(777000 + 1000 * op + files.index(fname))
                    c4 = [g for g in range(pf.num_row_groups) if g % 4 == op and g not in used]
                    used.update(r4.sample(c4, min(6, len(c4))))
            if a.fresh_round >= 6:   # fresh #5 used 4 parts x 6 row groups per file with seed 888000
                for op in range(4):
                    r5 = random.Random(888000 + 1000 * op + files.index(fname))
                    c5 = [g for g in range(pf.num_row_groups) if g % 4 == op and g not in used]
                    used.update(r5.sample(c5, min(6, len(c5))))
            if a.fresh_round >= 7:   # fresh #6 used 4 parts x 6 row groups per file with seed 999000 (same 'used' as then)
                snap, add6 = set(used), set()
                for op in range(4):
                    r6 = random.Random(999000 + 1000 * op + files.index(fname))
                    c6 = [g for g in range(pf.num_row_groups) if g % 4 == op and g not in snap]
                    add6.update(r6.sample(c6, min(6, len(c6))))
                used |= add6
            if a.fresh_round >= 8:   # fresh #7 used 4 parts x 6 row groups per file with seed 1111000 (same 'used' as then)
                snap7, add7 = set(used), set()
                for op in range(4):
                    r7 = random.Random(1111000 + 1000 * op + files.index(fname))
                    c7 = [g for g in range(pf.num_row_groups) if g % 4 == op and g not in snap7]
                    add7.update(r7.sample(c7, min(6, len(c7))))
                used |= add7
            rnd = random.Random({4: 777000, 5: 888000, 6: 999000, 7: 1111000, 8: 1222000}.get(a.fresh_round, 999000) + 1000 * a.part + files.index(fname))
            rgs = [g for g in range(pf.num_row_groups) if g % a.parts == a.part and g not in used]
        rgs = sorted(rnd.sample(rgs, min(a.rg_per_file, len(rgs))))
        for b in pf.iter_batches(batch_size=5000, columns=cols, row_groups=rgs):
            d = {c: b.column(c).to_pylist() for c in cols}
            for i in range(len(d["text"])):
                stats["seen"] += 1
                date = str(d.get("date", [""] * len(d["text"]))[i] or "")[:10]
                if not date or date >= CUTOFF:
                    stats["after_cutoff"] += 1
                    continue
                url = d["url"][i]
                ex = exclude_reason(url)
                if ex:
                    stats["excluded_" + ex] += 1
                    continue
                h = host_of(url)
                if a.fresh and not is_test_host(h):
                    continue
                key0 = str(d.get("id", [None] * len(d["text"]))[i] or url)
                raw.write(json.dumps({"doc_id": key0, "url": url, "host": h, "date": date,
                                      "dump": d.get("dump", [""] * len(d["text"]))[i], "text": d["text"][i]},
                                     ensure_ascii=False) + "\n")
                stats["raw_saved"] += 1
                if hosts[h] >= a.host_cap:
                    stats["host_cap"] += 1
                    continue
                key = str(d.get("id", [None] * len(d["text"]))[i] or url)
                seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
                if a.raw_test and not is_test_host(h):
                    continue
                p = raw_passage(d["text"][i]) if a.raw_test else clean_passage(d["text"][i], seed)
                if not p:
                    stats["no_passage"] += 1
                    continue
                if not ok_text(p):
                    stats["filtered"] += 1
                    continue
                hsh = hashlib.sha1(re.sub(r"\s+", " ", p).encode()).hexdigest()
                if hsh in seen:
                    stats["dup"] += 1
                    continue
                seen.add(hsh)
                hosts[h] += 1
                g = genre_v2(url, p)
                stats["genre_" + g] += 1
                out.write(json.dumps({"id": "fw2_" + hsh[:16], "doc_id": key, "url": url, "host": h, "genre": g, "date": date,
                                      "dump": d.get("dump", [""] * len(d["text"]))[i], "n_words": len(p.split()),
                                      "text": p}, ensure_ascii=False) + "\n")
                kept += 1
                stats["kept"] += 1
                if kept >= a.max_keep:
                    break
            if kept >= a.max_keep:
                break
        fh.close()
        print(fname, dict(stats), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    from pyrage import encrypt, x25519
    rcp = x25519.Recipient.from_str(open("recipient.txt").read().split()[0])
    os.makedirs("out", exist_ok=True)
    for name, b in (("passages", buf), ("raw", raw)):
        data = gzip.compress(b.getvalue().encode("utf-8"))
        open(f"out/fineweb_{('fresh_' if a.fresh_round == 4 else f'fresh{a.fresh_round}_') if a.fresh else ''}{name}_{a.part}.jsonl.gz.age", "wb").write(encrypt(data, [rcp]))
    json.dump({"part": a.part, "stats": stats, "n_hosts": len(hosts)}, open(f"out/status_{('fresh_' if a.fresh_round == 4 else f'fresh{a.fresh_round}_') if a.fresh else ''}{a.part}.json", "w"), indent=1)
    print("::notice::" + json.dumps(dict(stats))[:800], flush=True)


if __name__ == "__main__":
    main()
