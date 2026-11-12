"""
ZenESCO - Stage 1: data preparation and quality audit
=====================================================

Input  (made by convert_to_tables.py):  <BASE>/tables/
    ads.parquet, ad_text.parquet, ad_skills.parquet, skill_labels.parquet
Output: <BASE>/stage1/
    ads_clean.parquet          one row per ad, flags, duplicate groups, `analysable`
    ad_skills_clean.parquet    skill links of analysable ads only
    stage1_flow.csv            ads remaining after each cleaning step, by window (Table 1)
    stage1_portals.csv         portal x window counts, raw and analysable (metadata)
    boilerplate_candidates.csv skills that look like ad conventions (for inspection)
    top_titles.csv             most frequent normalised titles (analysable ads)
    title_label_sample.csv     300 ads to label by hand for Stage 2
    stage1_report.txt          readable summary of all of the above

Run:
    pip install pandas pyarrow numpy
    python stage1_prepare_data.py                 # uses C:\\skillab_data
    python stage1_prepare_data.py D:\\other\\base  # or another base folder

Python 3.8+. No ESCO files needed for this stage.
"""

import hashlib
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ============================================================
# CONFIGURATION (every value is reported in stage1_report.txt)
# ============================================================

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
TABLES = BASE / "tables"
OUT = BASE / "stage1"

PORTAL_ALIASES = {
    "jobbland.se": "jobbland",
    "lesjeudis.com": "lesjeudis",
    "jobbguru.se": "jobbguru",
    "jobmedic.co.uk": "jobmedic",
}

# Analysis windows (upload_date). Only the three ANALYSIS_WINDOWS are analysed.
WINDOWS = [
    ("pre",     None,         "2024-07-01"),
    ("H2_2024", "2024-07-01", "2025-01-01"),
    ("H1_2025", "2025-01-01", "2025-07-01"),
    ("gap",     "2025-07-01", "2025-12-01"),
    ("late",    "2025-12-01", "2026-09-01"),
    ("after",   "2026-09-01", None),
]
# The late window (Dec 2025 - Aug 2026) is not analysed: the first run showed that almost all
# of its ads have no extracted skills (7,365 raw -> 139 usable). It stays tagged and reported.
ANALYSIS_WINDOWS = ["H2_2024", "H1_2025"]

MIN_DESC_WORDS = 20          # shorter descriptions are not usable as evidence
SHINGLE = 5                  # word n-gram size for near-duplicate detection
MAX_WORDS = 3000             # words of each description used for MinHash (whole text in practice)
N_PERM = 64                  # MinHash permutations
BANDS, ROWS = 8, 8           # LSH banding (8 x 8 = 64); detection threshold ~0.77
NEAR_DUP_JACCARD = 0.90      # estimated Jaccard needed to call two ads near-duplicates
TEXT_BATCH = 5000            # rows of ad_text read at a time

# Boilerplate candidate screening (for inspection only; final rule in Stage 4)
BP_MIN_TITLE_ADS = 30        # a "frequent title" has at least this many analysable ads
BP_TITLE_PREV = 0.05         # skill counts as present in a title if in >= 5% of its ads
BP_MIN_PORTAL_PREV = 0.03    # candidate needs >= 3% prevalence in at least one portal
BP_MIN_TITLE_SHARE = 0.50    # ... and presence in >= 50% of frequent titles

LABEL_SAMPLE = 300
SEED = 20261026

# ============================================================
# HELPERS
# ============================================================

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


_GENDER = re.compile(
    r"[\(\[]?\s*\b(?:m|f|h|w|d|x|k|div)\s*/\s*(?:m|f|h|w|d|x|k|div)"
    r"(?:\s*/\s*(?:m|f|h|w|d|x|k|div))?\b\s*[\)\]]?", re.IGNORECASE)
_ALLGENDERS = re.compile(r"[\(\[]\s*(?:all genders|alla kön|tous genres)\s*[\)\]]", re.IGNORECASE)
_CONTRACT = re.compile(r"\b(?:cdi|cdd)\b", re.IGNORECASE)
_FILLER = re.compile(r"\bsökes\b", re.IGNORECASE)
_EDGE_PUNCT = re.compile(r"^[\s\-–—|,;:/.!*]+|[\s\-–—|,;:/.!*]+$")
_SPACES = re.compile(r"\s+")


def normalise_title(t):
    """Lower case, Unicode NFKC, strip gender markers, contract tags and filler words."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = _ALLGENDERS.sub(" ", t)
    t = _GENDER.sub(" ", t)
    t = _CONTRACT.sub(" ", t)
    t = _FILLER.sub(" ", t)
    t = t.replace("()", " ").replace("[]", " ")
    t = _SPACES.sub(" ", t)
    return _EDGE_PUNCT.sub("", t).strip()


def normalise_text(d):
    if d is None:
        return ""
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", str(d)).lower()).strip()


def window_of(dates):
    w = pd.Series("unknown", index=dates.index, dtype=object)
    for name, start, end in WINDOWS:
        m = dates.notna()
        if start:
            m &= dates >= pd.Timestamp(start)
        if end:
            m &= dates < pd.Timestamp(end)
        w[m] = name
    return w


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        root = i
        while self.p[root] != root:
            root = self.p[root]
        while self.p[i] != root:          # path compression
            self.p[i], i = root, self.p[i]
        return root

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)

    def roots(self):
        return np.array([self.find(i) for i in range(len(self.p))])


# MinHash with fixed random permutations (deterministic across runs)
_rng = np.random.RandomState(SEED)
_PRIME = np.uint64((1 << 61) - 1)
_A = _rng.randint(1, 2 ** 31 - 1, size=N_PERM).astype(np.uint64)
_B = _rng.randint(0, 2 ** 31 - 1, size=N_PERM).astype(np.uint64)
_EMPTY_SIG = np.full(N_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)


def minhash(words):
    if len(words) < SHINGLE:
        return None
    sh = {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    h = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
    return ((_A[:, None] * h[None, :] + _B[:, None]) % _PRIME).min(axis=1)


# ============================================================
# 1. LOAD ADS AND ASSIGN PORTAL, DATE, WINDOW, NORMALISED TITLE
# ============================================================

log(f"Reading {TABLES}")
OUT.mkdir(parents=True, exist_ok=True)
ads = pd.read_parquet(TABLES / "ads.parquet",
                      columns=["ad_id", "source", "title", "upload_date", "country",
                               "n_skills", "desc_words"])
ads["ad_id"] = ads["ad_id"].astype(str)
n_raw = len(ads)
ads["source_raw"] = ads["source"].astype(str)
ads["portal"] = ads["source_raw"].str.lower().map(lambda s: PORTAL_ALIASES.get(s, s))
ads["date"] = pd.to_datetime(ads["upload_date"].astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
ads["window"] = window_of(ads["date"])
ads["title_raw"] = ads["title"]
ads["title_norm"] = ads["title"].map(normalise_title)
ads = ads.drop(columns=["source", "title"]).reset_index(drop=True)
idx_of = pd.Series(ads.index.values, index=ads["ad_id"])
log(f"{n_raw:,} ads | {ads.portal.nunique()} portals after alias merge | "
    f"{ads.date.isna().sum():,} without a parsable date")

# ============================================================
# 2. ONE PASS OVER THE TEXT: EXACT-DUPLICATE KEYS AND MINHASH SIGNATURES
# ============================================================

n = len(ads)
exact_key = np.empty(n, dtype=object)
sigs = np.tile(_EMPTY_SIG, (n, 1))
has_sig = np.zeros(n, dtype=bool)
title_norm = ads["title_norm"].values

pf = pq.ParquetFile(TABLES / "ad_text.parquet")
done = 0
for batch in pf.iter_batches(batch_size=TEXT_BATCH, columns=["ad_id", "description"]):
    b = batch.to_pandas()
    for aid, desc in zip(b["ad_id"].astype(str), b["description"]):
        i = idx_of.get(aid)
        if i is None:
            continue
        d = normalise_text(desc)
        exact_key[i] = hashlib.md5((title_norm[i] + "||" + d).encode("utf-8")).hexdigest()
        s = minhash(d.split()[:MAX_WORDS])
        if s is not None:
            sigs[i] = s
            has_sig[i] = True
    done += len(b)
    if done % 50000 < TEXT_BATCH:
        log(f"  text pass: {done:,} ads")
log(f"text pass done: {has_sig.sum():,} ads with a MinHash signature")

# ============================================================
# 3. EXACT AND NEAR DUPLICATES
# ============================================================

uf_exact = UnionFind(n)
first_with_key = {}
for i, k in enumerate(exact_key):
    if k is None:
        continue
    j = first_with_key.setdefault(k, i)
    if j != i:
        uf_exact.union(i, j)
exact_root = uf_exact.roots()
log(f"exact duplicate groups: {(pd.Series(exact_root).value_counts() > 1).sum():,}")

uf_all = UnionFind(n)
for i, r in enumerate(exact_root):
    if r != i:
        uf_all.union(i, r)

# LSH candidates, verified by estimated Jaccard against the first member of each bucket.
# The bucket key includes the normalised title, so only ads with the SAME job title can be
# near duplicates: two different jobs from one employer share company boilerplate but not a
# title, and must not be merged (this also prevents chaining across different jobs).
cand_checked = near_links = 0
reps = np.where(has_sig & (exact_root == np.arange(n)) & (title_norm != ""))[0]
for band in range(BANDS):
    buckets = {}
    block = sigs[reps, band * ROWS:(band + 1) * ROWS]
    for pos, i in enumerate(reps):
        key = title_norm[i].encode("utf-8") + b"||" + block[pos].tobytes()
        first = buckets.get(key)
        if first is None:
            buckets[key] = i
            continue
        cand_checked += 1
        if np.mean(sigs[i] == sigs[first]) >= NEAR_DUP_JACCARD:
            uf_all.union(i, first)
            near_links += 1
    log(f"  LSH band {band + 1}/{BANDS}: {cand_checked:,} candidate pairs checked so far")
all_root = uf_all.roots()

ads["exact_group"] = ads["ad_id"].values[exact_root]
ads["dup_group"] = ads["ad_id"].values[all_root]
ads["dup_size"] = ads.groupby("dup_group")["ad_id"].transform("size")


def keep_earliest(group_col):
    order = ads.assign(_d=ads["date"].fillna(pd.Timestamp("2100-01-01"))) \
               .sort_values(["_d", "ad_id"])
    keep_ids = set(order.drop_duplicates(group_col)["ad_id"])
    return ads["ad_id"].isin(keep_ids)


ads["kept_exact"] = keep_earliest("exact_group")
ads["kept"] = keep_earliest("dup_group")
log(f"after exact de-duplication: {ads.kept_exact.sum():,} | after near de-duplication: {ads.kept.sum():,}")

# ============================================================
# 4. USABILITY FLAGS AND THE ANALYSABLE SET
# ============================================================

ads["empty_title"] = ads["title_norm"].eq("")
ads["short_desc"] = ads["desc_words"].fillna(0) < MIN_DESC_WORDS
ads["zero_skills"] = ads["n_skills"].fillna(0) == 0
ads["in_window"] = ads["window"].isin(ANALYSIS_WINDOWS)
ads["analysable"] = ads.kept & ads.in_window & ~ads.empty_title & ~ads.short_desc & ~ads.zero_skills
log(f"analysable ads: {ads.analysable.sum():,}")

# Flow table (Table 1): cumulative steps, by analysis window
steps = [
    ("1 raw ads in analysis windows", ads.in_window),
    ("2 after exact de-duplication", ads.in_window & ads.kept_exact),
    ("3 after near de-duplication", ads.in_window & ads.kept),
    ("4 after removing empty titles", ads.in_window & ads.kept & ~ads.empty_title),
    ("5 after removing short descriptions", ads.in_window & ads.kept & ~ads.empty_title & ~ads.short_desc),
    ("6 after removing ads without skills (analysable)", ads.analysable),
]
flow = pd.DataFrame([{"step": s, "total": int(m.sum()),
                      **{w: int((m & (ads.window == w)).sum()) for w in ANALYSIS_WINDOWS}}
                     for s, m in steps])
flow.to_csv(OUT / "stage1_flow.csv", index=False)

windows_raw = ads["window"].value_counts().reindex([w for w, _, _ in WINDOWS] + ["unknown"]).fillna(0).astype(int)

portals = (ads.groupby(["portal", "window"]).agg(raw=("ad_id", "size"), analysable=("analysable", "sum"))
              .reset_index())
portal_country = (ads.groupby("portal")["country"]
                     .agg(lambda s: s.value_counts().index[0] if s.notna().any() else None)
                     .rename("main_country_metadata"))
portals = portals.merge(portal_country, on="portal", how="left")
portals.to_csv(OUT / "stage1_portals.csv", index=False)

# ============================================================
# 5. SKILLS OF ANALYSABLE ADS AND BOILERPLATE CANDIDATES
# ============================================================

log("reading ad_skills")
ad_skills = pd.read_parquet(TABLES / "ad_skills.parquet")
ad_skills["ad_id"] = ad_skills["ad_id"].astype(str)
ad_skills = ad_skills.drop_duplicates(["ad_id", "skill_uri"])
ana = ads.loc[ads.analysable, ["ad_id", "portal", "window", "title_norm"]]
sk = ad_skills.merge(ana, on="ad_id", how="inner")
sk[["ad_id", "skill_uri"]].to_parquet(OUT / "ad_skills_clean.parquet", index=False)
labels = pd.read_parquet(TABLES / "skill_labels.parquet").drop_duplicates("skill_uri") \
           .set_index("skill_uri")["label"]
N_ana = len(ana)
log(f"{len(sk):,} skill links on {N_ana:,} analysable ads")

overall = sk.groupby("skill_uri")["ad_id"].nunique().rename("ads").to_frame()
overall["prevalence"] = overall["ads"] / N_ana

portal_n = ana.groupby("portal").size()
pp = sk.groupby(["skill_uri", "portal"])["ad_id"].nunique().rename("n").reset_index()
pp["prev"] = pp["n"] / pp["portal"].map(portal_n)
top_portal = pp.sort_values("prev", ascending=False).drop_duplicates("skill_uri") \
               .set_index("skill_uri")[["portal", "prev"]] \
               .rename(columns={"portal": "top_portal", "prev": "top_portal_prev"})

title_n = ana.groupby("title_norm").size()
freq_titles = title_n[title_n >= BP_MIN_TITLE_ADS]
tf = sk[sk.title_norm.isin(freq_titles.index)]
tp = tf.groupby(["skill_uri", "title_norm"])["ad_id"].nunique().rename("n").reset_index()
tp["prev"] = tp["n"] / tp["title_norm"].map(freq_titles)
spread = tp[tp.prev >= BP_TITLE_PREV].groupby("skill_uri").size().rename("frequent_titles_present")
n_ft = max(len(freq_titles), 1)

bp = overall.join(top_portal).join(spread).fillna({"frequent_titles_present": 0})
bp["share_frequent_titles"] = bp["frequent_titles_present"] / n_ft
bp["label"] = bp.index.map(labels)
bp["candidate"] = (bp.top_portal_prev >= BP_MIN_PORTAL_PREV) & (bp.share_frequent_titles >= BP_MIN_TITLE_SHARE)
bp = bp.sort_values(["candidate", "share_frequent_titles", "prevalence"], ascending=False)
bp = bp.reset_index()[["skill_uri", "label", "ads", "prevalence", "top_portal", "top_portal_prev",
                       "frequent_titles_present", "share_frequent_titles", "candidate"]]
bp.to_csv(OUT / "boilerplate_candidates.csv", index=False)
log(f"boilerplate candidates: {int(bp.candidate.sum())} skills "
    f"(over {len(freq_titles)} frequent titles)")

# ============================================================
# 6. TOP TITLES AND THE HAND-LABELLING SAMPLE FOR STAGE 2
# ============================================================

top_titles = title_n.sort_values(ascending=False).head(500).rename_axis("title_norm").reset_index(name="ads")
top_titles.to_csv(OUT / "top_titles.csv", index=False)

rng = np.random.RandomState(SEED)
share = portal_n / portal_n.sum()
alloc = (share * LABEL_SAMPLE).round().astype(int).clip(lower=10)
alloc = alloc.clip(upper=portal_n)
picks = []
for p, k in alloc.items():
    pool = ads.loc[ads.analysable & (ads.portal == p), "ad_id"].values
    picks.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
sample = ads.loc[ads.ad_id.isin(picks), ["ad_id", "portal", "window", "title_raw", "title_norm"]].copy()

snippets = {}
want = set(sample.ad_id)
for batch in pf.iter_batches(batch_size=TEXT_BATCH, columns=["ad_id", "description"]):
    b = batch.to_pandas()
    b = b[b["ad_id"].astype(str).isin(want)]
    for aid, desc in zip(b["ad_id"].astype(str), b["description"]):
        snippets[aid] = " ".join(str(desc or "").split()[:60])
sample["description_start"] = sample["ad_id"].map(snippets)
sample["esco_occupation_label"] = ""
sample["esco_occupation_uri"] = ""
sample["notes"] = ""
sample = sample.sample(frac=1, random_state=SEED)
sample.to_csv(OUT / "title_label_sample.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 7. SAVE ADS AND WRITE THE REPORT
# ============================================================

cols = ["ad_id", "portal", "source_raw", "country", "upload_date", "window", "title_raw", "title_norm",
        "desc_words", "n_skills", "exact_group", "dup_group", "dup_size", "kept_exact", "kept",
        "empty_title", "short_desc", "zero_skills", "in_window", "analysable"]
ads[cols].to_parquet(OUT / "ads_clean.parquet", index=False)

dup_groups = ads[ads.dup_size > 1]
lines = []
w = lines.append
w("ZenESCO - Stage 1 report: data preparation and quality audit")
w("=" * 70)
w(f"Input: {TABLES}")
w(f"Raw ads: {n_raw:,} | portals after alias merge: {ads.portal.nunique()}")
w("")
w("Configuration")
for k in ["MIN_DESC_WORDS", "SHINGLE", "MAX_WORDS", "N_PERM", "BANDS", "ROWS", "NEAR_DUP_JACCARD",
          "BP_MIN_TITLE_ADS", "BP_TITLE_PREV", "BP_MIN_PORTAL_PREV", "BP_MIN_TITLE_SHARE", "SEED"]:
    w(f"  {k} = {globals()[k]}")
w("")
w(f"Raw ads per window (all windows; analysed: {', '.join(ANALYSIS_WINDOWS)})")
w(windows_raw.to_string())
w("")
w("Cleaning flow (Table 1)")
w(flow.to_string(index=False))
w("")
w("Duplicates")
w(f"  ads in a duplicate group (size > 1): {len(dup_groups):,}")
w(f"  removed as exact duplicates:          {int((~ads.kept_exact).sum()):,}")
w(f"  removed as near duplicates (extra):   {int((ads.kept_exact & ~ads.kept).sum()):,}")
w(f"  LSH candidate pairs checked: {cand_checked:,}; near-duplicate links: {near_links:,}")
w(f"  largest duplicate group: {int(ads.dup_size.max())} ads")
g = (dup_groups.groupby("dup_group")
     .agg(size=("ad_id", "size"), titles=("title_norm", "nunique"),
          portals=("portal", lambda s: ", ".join(sorted(s.unique()))),
          title=("title_norm", lambda s: s.value_counts().index[0]),
          first=("date", "min"), last=("date", "max"))
     .sort_values("size", ascending=False))
w(f"  duplicate groups: {len(g):,}; groups with more than one title: {int((g.titles > 1).sum()):,} "
  f"(should be 0 with title-gated near duplicates)")
w(f"  group size distribution: {g['size'].describe(percentiles=[.5, .9, .99]).round(1).to_dict()}")
w("  10 largest groups (a reposted job keeps its title; check that these are plausible reposts):")
top_g = g.head(10).copy()
top_g["first"] = top_g["first"].dt.date
top_g["last"] = top_g["last"].dt.date
w(top_g[["size", "titles", "title", "portals", "first", "last"]].to_string(index=False))
w("")
w("Share of ads WITHOUT extracted skills, by window and portal (all ads, before cleaning)")
zs = ads.pivot_table(index="portal", columns="window", values="zero_skills", aggfunc="mean")
w((zs.reindex(columns=[c for c, _, _ in WINDOWS if c in zs.columns]) * 100).round(1).to_string())
w("")
w("Usability flags among kept ads in analysis windows")
kw = ads.kept & ads.in_window
w(f"  empty title:        {int((kw & ads.empty_title).sum()):,}")
w(f"  short description:  {int((kw & ads.short_desc).sum()):,}")
w(f"  zero skills:        {int((kw & ads.zero_skills).sum()):,}")
w("")
w("Analysable ads per portal (country shown as metadata only)")
pt = portals.pivot_table(index=["portal", "main_country_metadata"], columns="window",
                         values="analysable", aggfunc="sum", fill_value=0)
w(pt.reindex(columns=[c for c in ANALYSIS_WINDOWS if c in pt.columns]).to_string())
w("")
w(f"Skills on analysable ads: {sk.skill_uri.nunique():,} distinct, {len(sk):,} links, "
  f"{len(sk) / max(N_ana, 1):.1f} per ad")
w("")
w(f"Boilerplate candidates ({int(bp.candidate.sum())}); top 30 by spread over frequent titles")
w(bp[bp.candidate].head(30)[["label", "prevalence", "top_portal", "top_portal_prev",
                             "share_frequent_titles"]].round(3).to_string(index=False))
w("")
w("Most common skills overall (top 30), for comparison")
w(bp.sort_values("prevalence", ascending=False).head(30)[
    ["label", "prevalence", "share_frequent_titles", "candidate"]].round(3).to_string(index=False))
w("")
w(f"Distinct normalised titles among analysable ads: {title_n.size:,}; "
  f"frequent titles (>= {BP_MIN_TITLE_ADS} ads): {len(freq_titles):,}")
w("Top 40 titles")
w(top_titles.head(40).to_string(index=False))
w("")
w(f"Title labelling sample: {len(sample)} ads -> title_label_sample.csv")
w(f"Runtime: {time.time() - T0:.0f}s")
(OUT / "stage1_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
log(f"all outputs written to {OUT}")
