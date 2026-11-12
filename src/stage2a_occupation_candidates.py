"""
ZenESCO - Stage 2a: occupation candidates from job titles
=========================================================

Gives every analysable ad (from Stage 1) its most likely ESCO occupations, using the
job TITLE ONLY (never the skills, so occupation and skill evidence stay independent).

  1. Exact match: the normalised title equals an ESCO occupation's PREFERRED label
     (English ESCO v1.2). Matches on alternative/hidden labels are recorded but not trusted
     blindly, because some are misleading (e.g. "maintenance technician" -> welder).
  2. Otherwise: nearest ESCO labels with a multilingual sentence-embedding model,
     so Swedish / French / German / Greek titles can match English ESCO labels.
     Top-3 candidate occupations with similarity scores are stored, plus a second opinion:
     the best match among PREFERRED labels only (pref_occ, pref_score).

The acceptance threshold is NOT chosen here. It is calibrated in Stage 2b against the
hand-labelled sample (title_label_sample.csv), so this script only proposes candidates.

Input:
    <BASE>/stage1/ads_clean.parquet, <BASE>/stage1/ad_skills_clean.parquet
    <BASE>/esco/  ESCO v1.2 CSV files (names may carry a prefix; matched by ending):
        occupations_en.csv, skills_en.csv, occupationSkillRelations_en.csv,
        ISCOGroups_en.csv, languageSkillsCollection_en.csv
Output: <BASE>/stage2/
    title_candidates.parquet      one row per distinct title: method, top-3 occupations, scores
    ad_occupation_candidates.parquet  one row per analysable ad with its title's candidates
    stage2a_report.txt            coverage, score distribution, provisional counts per occupation

Run:
    pip install pandas pyarrow numpy sentence-transformers
    python stage2a_occupation_candidates.py              # BASE = C:\\skillab_data
    python stage2a_occupation_candidates.py D:\\other    # other base folder
The first run downloads the embedding model (~470 MB) once. On a laptop CPU, embedding
~70,000 titles takes roughly 10-30 minutes; embeddings are cached for later runs.
"""

import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
ESCO_DIR = BASE / "esco"
STAGE1 = BASE / "stage1"
OUT = BASE / "stage2"

ENCODER = "sbert"   # "sbert" (multilingual model, default) or "tfidf" (fallback, no download)
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH = 256
TOP_LABELS = 30            # labels retrieved per title before collapsing to occupations
PROVISIONAL_THRESHOLDS = [0.60, 0.70, 0.80]   # only for the provisional counts in the report

# 65 target ESCO occupations in 12 clusters (same list as the crawler)
TARGETS = {1: {'label': 'Industrial and manufacturing engineering',
     'uris': ['http://data.europa.eu/esco/occupation/53d2052f-edc0-4f44-8be1-164526412bfe',
              'http://data.europa.eu/esco/occupation/6818c837-072a-4120-b913-bd360b7b14d0',
              'http://data.europa.eu/esco/occupation/d7380260-1386-4e36-98b5-8115ca4a3536',
              'http://data.europa.eu/esco/occupation/b5eaf231-77ad-4a86-8e54-15cc4398aad2',
              'http://data.europa.eu/esco/occupation/bcc21c63-7eee-4520-8fa7-43eefd389668',
              'http://data.europa.eu/esco/occupation/53be4843-8f31-4645-b25c-9a5bbb04f6b6',
              'http://data.europa.eu/esco/occupation/11833a96-fc27-4b03-a53d-892ac891a2bb']},
 2: {'label': 'Automation, instrumentation and control',
     'uris': ['http://data.europa.eu/esco/occupation/bb609566-3ab6-44dd-8f48-cf0b15b96827',
              'http://data.europa.eu/esco/occupation/5791a961-4717-4230-a245-a4082a675f04',
              'http://data.europa.eu/esco/occupation/e6800524-eec2-4c33-af0e-6f36d1d679bc',
              'http://data.europa.eu/esco/occupation/2bb8adee-2b84-4d7b-84f9-0620bc09ada8',
              'http://data.europa.eu/esco/occupation/76abbb82-c103-4d7a-a4c0-14dba4d6199a',
              'http://data.europa.eu/esco/occupation/08f5481e-9233-4ef2-9e4b-945224879ce5']},
 3: {'label': 'Robotics and mechatronics',
     'uris': ['http://data.europa.eu/esco/occupation/551a6797-3ca9-41fc-9dca-9a982a8fd8a6',
              'http://data.europa.eu/esco/occupation/7833d5cd-873d-4fdd-b2f8-9762d68494a7',
              'http://data.europa.eu/esco/occupation/ab360abd-32e2-4e03-967d-a10758efffa7',
              'http://data.europa.eu/esco/occupation/edf2e989-d7c5-496e-b365-81fc5cb9eb39',
              'http://data.europa.eu/esco/occupation/5001506f-dd7d-4801-92ac-6f7e93b45133',
              'http://data.europa.eu/esco/occupation/869e3a85-6e4c-4e3c-b14f-a8399c1abd2d']},
 4: {'label': 'Maintenance and reliability',
     'uris': ['http://data.europa.eu/esco/occupation/615920c5-4f63-4eb3-8b60-afaaed3ab1ff',
              'http://data.europa.eu/esco/occupation/8edd8bff-cb59-4c9c-ab0f-59e77d18be48',
              'http://data.europa.eu/esco/occupation/a9b20949-897b-481f-8642-d1f69263baff',
              'http://data.europa.eu/esco/occupation/07dfff6d-a0d0-4202-a697-fe9176c15999']},
 5: {'label': 'Quality, testing and metrology',
     'uris': ['http://data.europa.eu/esco/occupation/30c0943c-caa1-411d-b273-26f475971273',
              'http://data.europa.eu/esco/occupation/998a0bf4-b1be-42ab-aee5-0096c9be3319',
              'http://data.europa.eu/esco/occupation/c8fa93eb-7c2c-42c3-b135-c2e825a6615e',
              'http://data.europa.eu/esco/occupation/fdb16700-109e-4d02-949e-3213f278df93',
              'http://data.europa.eu/esco/occupation/c332af81-4481-4494-9786-aea1f5c5ffcf',
              'http://data.europa.eu/esco/occupation/e6a3ce2e-c0f2-47a7-937a-61ae9abfe1fa']},
 6: {'label': 'Smart manufacturing and electrical/electronic engineering',
     'uris': ['http://data.europa.eu/esco/occupation/82616669-cc82-4455-9332-a2df209e9f3e',
              'http://data.europa.eu/esco/occupation/418f3a43-a6b2-4101-aaa0-832fe7d82885',
              'http://data.europa.eu/esco/occupation/4f4d33a1-0b41-4dd9-8a42-aeba5a3974ab',
              'http://data.europa.eu/esco/occupation/4d34a9ee-2653-4da0-bef7-f59294577224',
              'http://data.europa.eu/esco/occupation/77abfaec-a250-4765-95fa-6091e8da1bba',
              'http://data.europa.eu/esco/occupation/86ca306c-ab99-420a-9e2a-aa73c5c4de22']},
 7: {'label': 'Product design and development engineering',
     'uris': ['http://data.europa.eu/esco/occupation/6fc8f605-98b9-4218-b5e2-91c4c9c55c4d',
              'http://data.europa.eu/esco/occupation/579254cf-6d69-4889-9000-9c79dc568644',
              'http://data.europa.eu/esco/occupation/79fed799-ab3a-43d9-bd91-414c2c3b2f57',
              'http://data.europa.eu/esco/occupation/42381741-f3a7-4b26-b35c-da6fad9f1cfd',
              'http://data.europa.eu/esco/occupation/ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775',
              'http://data.europa.eu/esco/occupation/fde4372d-8293-42de-afbb-8b5a9b274fe7']},
 8: {'label': 'Engineering data, CAD and technical documentation',
     'uris': ['http://data.europa.eu/esco/occupation/62979364-6fac-41f2-87ca-202bca19a6ab',
              'http://data.europa.eu/esco/occupation/a4869ae9-2dfe-48cc-bd97-f803880f3407',
              'http://data.europa.eu/esco/occupation/8cb59bc4-1f39-477e-beea-effed12d186c',
              'http://data.europa.eu/esco/occupation/1f958638-066f-470c-9549-a9070ba280a6']},
 9: {'label': 'Production management and planning',
     'uris': ['http://data.europa.eu/esco/occupation/6426ada1-4d8c-4612-99e3-91bc569805ee',
              'http://data.europa.eu/esco/occupation/eb9479c6-5fdf-4f45-ace4-28b547b8c7e5',
              'http://data.europa.eu/esco/occupation/c6bd511a-d966-4df9-a48e-4f800354f268',
              'http://data.europa.eu/esco/occupation/40517ed4-2769-4283-a15b-4b0d939bd2d3',
              'http://data.europa.eu/esco/occupation/b12b14fd-5b74-40c6-afd9-dd35b34e27ed',
              'http://data.europa.eu/esco/occupation/c9862fdd-53b7-4da9-b742-4302d704836c',
              'http://data.europa.eu/esco/occupation/219d1c30-3849-4de3-b0d6-07da187b6521']},
 10: {'label': 'Supply chain planning',
      'uris': ['http://data.europa.eu/esco/occupation/aacc3918-b5d3-484b-9480-5d29aa550d74',
               'http://data.europa.eu/esco/occupation/aeb7c6bd-22c9-43c2-8551-a1cc2560584f',
               'http://data.europa.eu/esco/occupation/dbdc9f7b-31c6-4f21-a9c5-48253bb69c08',
               'http://data.europa.eu/esco/occupation/51cd3a7a-eae6-4dc6-9f24-179d7ec6d230',
               'http://data.europa.eu/esco/occupation/c8b296be-54ae-48cf-be21-031fa8457937']},
 11: {'label': 'Logistics and distribution',
      'uris': ['http://data.europa.eu/esco/occupation/c4071145-a774-4cf4-aefa-ae18a385cca4',
               'http://data.europa.eu/esco/occupation/823db06f-25e1-4658-b063-d6ef956b41f5',
               'http://data.europa.eu/esco/occupation/a2ff6cc3-15fa-4f02-a457-8fd5df7aa29e',
               'http://data.europa.eu/esco/occupation/2f5de1ab-eddf-4715-8193-e994912c22ea',
               'http://data.europa.eu/esco/occupation/632e61c8-4e4b-44c7-b213-6dd5a174b143']},
 12: {'label': 'Purchasing and procurement',
      'uris': ['http://data.europa.eu/esco/occupation/377865f0-c327-4599-8c7e-d428c6588edf',
               'http://data.europa.eu/esco/occupation/29fa406d-de89-4515-be8d-0fe7bda993f1',
               'http://data.europa.eu/esco/occupation/c9b2b0a7-0adc-4ca7-bf7c-dc95e18aeac7']}}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def esco_file(name):
    """Find an ESCO file by its official name, also when the name carries a prefix."""
    hits = [p for p in ESCO_DIR.glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    if not hits:
        sys.exit(f"ESCO file '{name}' not found in {ESCO_DIR}")
    return hits[0]


def key(u):
    return str(u).rstrip("/").split("/")[-1].lower()


# ============================================================
# TEXT NORMALISATION (same spirit as Stage 1, plus a "core title")
# ============================================================

_SPACES = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s/&+.#-]", re.UNICODE)
_SPLIT = re.compile(r"\s(?:-|–|—|\||@)\s|,|\(|\[|:")
_SENIORITY = re.compile(
    r"\b(?:senior|sr|junior|jr|lead|principal|chief|head of|trainee|intern|stagiaire|"
    r"erfaren|erfarna|ung|nyexaminerad|confirmé|confirmée|débutant|expérimenté|expérimentée|"
    r"i|ii|iii|iv)\b\.?", re.IGNORECASE)


def norm(t):
    t = unicodedata.normalize("NFKC", str(t or "")).lower()
    t = _PUNCT.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def core_title(t):
    """Title without seniority words and without trailing location/company parts."""
    t = str(t or "")
    t = _SPLIT.split(t)[0]
    t = _SENIORITY.sub(" ", t)
    return norm(t)


# ============================================================
# ENCODERS
# ============================================================

def make_encoder():
    if ENCODER == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL)

        def enc(texts):
            return model.encode(list(texts), batch_size=BATCH, show_progress_bar=True,
                                normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        return enc, SBERT_MODEL
    # fallback: character n-gram TF-IDF (works across close cognates, weaker across languages)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = {}

    def enc(texts, fit=None):
        if "v" not in vec:
            vec["v"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                       sublinear_tf=True).fit(fit if fit is not None else texts)
        m = vec["v"].transform(list(texts)).astype(np.float32)
        return m
    return enc, "tfidf-char-3-5"


# ============================================================
# 1. LOAD
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)
log(f"reading Stage 1 outputs from {STAGE1}")
ads = pd.read_parquet(STAGE1 / "ads_clean.parquet",
                      columns=["ad_id", "portal", "window", "title_raw", "title_norm", "analysable"])
ads = ads[ads.analysable].drop(columns="analysable").reset_index(drop=True)
log(f"{len(ads):,} analysable ads")

log(f"reading ESCO files from {ESCO_DIR}")
occ = pd.read_csv(esco_file("occupations_en.csv"))
skills = pd.read_csv(esco_file("skills_en.csv"), usecols=["conceptUri", "preferredLabel"])
isco = pd.read_csv(esco_file("ISCOGroups_en.csv"), usecols=["code", "preferredLabel"])
lang = pd.read_csv(esco_file("languageSkillsCollection_en.csv"), usecols=["conceptUri"])
occ["iscoGroup"] = occ["iscoGroup"].astype(str).str.zfill(4)
isco["code"] = isco["code"].astype(str)
isco_label = dict(zip(isco.code, isco.preferredLabel))
occ_label = dict(zip(occ.conceptUri, occ.preferredLabel))
occ_isco = dict(zip(occ.conceptUri, occ.iscoGroup))
log(f"ESCO: {len(occ):,} occupations, {len(skills):,} skills")

# ESCO labels -> occupation (preferred, alternative, hidden; labels separated by newlines)
rows = []
for _, r in occ.iterrows():
    rows.append((r.conceptUri, r.preferredLabel, "preferred"))
    for col, kind in (("altLabels", "alternative"), ("hiddenLabels", "hidden")):
        if isinstance(r[col], str):
            for lab in re.split(r"\n|\|", r[col]):
                if lab.strip():
                    rows.append((r.conceptUri, lab.strip(), kind))
labels = pd.DataFrame(rows, columns=["occ_uri", "label", "kind"])
labels["label_norm"] = labels["label"].map(norm)
labels = labels[labels.label_norm != ""].drop_duplicates(["occ_uri", "label_norm"]).reset_index(drop=True)
log(f"{len(labels):,} ESCO occupation labels")

# ============================================================
# 2. DISTINCT TITLES, EXACT MATCHES
# ============================================================

titles = pd.DataFrame({"title_norm": ads["title_norm"].unique()})
titles["core"] = titles["title_norm"].map(core_title)
titles.loc[titles.core == "", "core"] = titles["title_norm"].map(norm)
log(f"{len(titles):,} distinct titles, {titles.core.nunique():,} distinct core titles")

# Exact matches are split by label kind. A match on a PREFERRED label is reliable; a match on
# an alternative/hidden label is not always (e.g. in ESCO v1.2 "maintenance technician" is an
# alternative label of "welder"), so its accuracy is measured separately in Stage 2b.
pref_map = labels[labels.kind == "preferred"].groupby("label_norm")["occ_uri"].first()
alt_map = labels[labels.kind != "preferred"].groupby("label_norm")["occ_uri"].apply(list)

cores = pd.DataFrame({"core": titles["core"].unique()})
cores["exact_pref"] = cores["core"].map(pref_map)
cores["exact_alt"] = cores["core"].map(lambda c: alt_map.get(c, [None])[0] if c in alt_map.index else None)
cores["exact_alt_n"] = cores["core"].map(lambda c: len(set(alt_map.get(c, []))) if c in alt_map.index else 0)
log(f"exact matches: preferred label {cores.exact_pref.notna().sum():,}, "
    f"alternative/hidden label only {(cores.exact_pref.isna() & cores.exact_alt.notna()).sum():,}, "
    f"of {len(cores):,} core titles")

# ============================================================
# 3. EMBEDDING CANDIDATES FOR ALL CORE TITLES
# ============================================================

encode, enc_name = make_encoder()
cache = OUT / f"embeddings_{re.sub(r'[^A-Za-z0-9]+', '_', enc_name)}.npz"
lab_texts = labels["label"].tolist()
core_texts = cores["core"].tolist()

if ENCODER == "sbert":
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["labels"]) == lab_texts and list(z["cores"]) == core_texts:
            L, C = z["L"], z["C"]
            log("embeddings loaded from cache")
        else:
            cache.unlink()
    if not cache.exists():
        log(f"encoding {len(lab_texts):,} ESCO labels with {enc_name}")
        L = encode(lab_texts)
        log(f"encoding {len(core_texts):,} core titles")
        C = encode(core_texts)
        np.savez(cache, L=L, C=C, labels=np.array(lab_texts, dtype=object),
                 cores=np.array(core_texts, dtype=object))
else:
    L = encode(lab_texts, fit=lab_texts + core_texts)
    C = encode(core_texts)

lab_occ = labels["occ_uri"].values
is_pref_lab = (labels["kind"] == "preferred").values
cand = [[None] * 3 for _ in range(len(cores))]
score = np.zeros((len(cores), 3), dtype=np.float32)
pref_occ = [None] * len(cores)          # best match among PREFERRED labels only (second opinion)
pref_score = np.zeros(len(cores), dtype=np.float32)
pref_idx = np.where(is_pref_lab)[0]
step = 2000
for s in range(0, len(cores), step):
    sims = C[s:s + step] @ L.T
    sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
    sp = sims[:, pref_idx]
    bp = sp.argmax(axis=1)
    for r in range(sims.shape[0]):
        pref_occ[s + r] = lab_occ[pref_idx[bp[r]]]
        pref_score[s + r] = sp[r, bp[r]]
    top = np.argpartition(-sims, kth=min(TOP_LABELS, sims.shape[1] - 1), axis=1)[:, :TOP_LABELS]
    for r in range(sims.shape[0]):
        order = top[r][np.argsort(-sims[r, top[r]])]
        seen = []
        for j in order:
            u = lab_occ[j]
            if u not in seen:
                seen.append(u)
                k = len(seen) - 1
                cand[s + r][k] = u
                score[s + r, k] = sims[r, j]
                if len(seen) == 3:
                    break
    log(f"  candidates: {min(s + step, len(cores)):,}/{len(cores):,} core titles")

for k in range(3):
    cores[f"occ{k + 1}"] = [c[k] for c in cand]
    cores[f"score{k + 1}"] = score[:, k]
cores["pref_occ"] = pref_occ
cores["pref_score"] = pref_score

# Keep the embedding's own first choice, then let a preferred-label match override it.
# Alternative-label matches are recorded (exact_alt) but do NOT override; Stage 2b decides.
cores["emb_occ1"] = cores["occ1"]
cores["emb_score1"] = cores["score1"]
is_pref = cores.exact_pref.notna()
cores["method"] = np.where(is_pref, "exact_preferred",
                           np.where(cores.exact_alt.notna(), "exact_alternative", "embedding"))
cores.loc[is_pref, "occ1"] = cores.loc[is_pref, "exact_pref"]
cores.loc[is_pref, "score1"] = 1.0
cores["margin"] = cores["score1"] - cores["score2"]
cores["occ1_label"] = cores["occ1"].map(occ_label)
cores["occ1_isco"] = cores["occ1"].map(occ_isco)
cores["exact_alt_label"] = cores["exact_alt"].map(occ_label)
cores["pref_label"] = cores["pref_occ"].map(occ_label)

titles = titles.merge(cores, on="core", how="left")
titles.to_parquet(OUT / "title_candidates.parquet", index=False)

ad_c = ads.merge(titles, on="title_norm", how="left")
ad_c.to_parquet(OUT / "ad_occupation_candidates.parquet", index=False)
log(f"wrote {OUT / 'ad_occupation_candidates.parquet'}")

# ============================================================
# 4. CHECKS AND REPORT
# ============================================================

# Tracker skill URIs vs ESCO v1.2 skills (version alignment)
ask = pd.read_parquet(STAGE1 / "ad_skills_clean.parquet")
esco_sk = set(skills.conceptUri.map(key))
lang_sk = set(lang.conceptUri.map(key))
uniq = pd.Series(ask.skill_uri.unique())
in_esco = uniq.map(key).isin(esco_sk)
link_in = ask.skill_uri.map(key).isin(esco_sk)
link_lang = ask.skill_uri.map(key).isin(lang_sk)

target_uris = {u: (n, t["label"]) for n, t in TARGETS.items() for u in t["uris"]}
missing_t = [u for u in target_uris if u not in occ_label]

lines = []
w = lines.append
w("ZenESCO - Stage 2a report: occupation candidates from job titles")
w("=" * 70)
w(f"Encoder: {enc_name} | ESCO occupations: {len(occ):,} | labels: {len(labels):,}")
w(f"Analysable ads: {len(ads):,} | distinct titles: {len(titles):,} | distinct core titles: {len(cores):,}")
w("")
w("Version check: Tracker skill URIs vs ESCO v1.2 skills")
w(f"  distinct skill URIs in ads: {len(uniq):,}; found in ESCO skills_en: {in_esco.mean() * 100:.1f}%")
w(f"  ad-skill links pointing to an ESCO skill: {link_in.mean() * 100:.1f}%")
w(f"  ad-skill links that are ESCO language skills: {link_lang.mean() * 100:.1f}%")
if (~in_esco).any():
    w(f"  examples of skill URIs not in ESCO v1.2: {list(uniq[~in_esco].head(5))}")
w("")
w(f"Target occupations: {len(target_uris)} from the crawler list; missing in ESCO v1.2: {len(missing_t)}")
w("")
ad_m = ad_c["method"].value_counts()
w("How ads got their first candidate")
for m_ in ["exact_preferred", "exact_alternative", "embedding"]:
    w(f"  {m_:18s} {int(ad_m.get(m_, 0)):>8,} ads ({ad_m.get(m_, 0) / len(ad_c) * 100:.1f}%)")
alt = ad_c[ad_c.method == "exact_alternative"]
agree = (alt.exact_alt == alt.occ1).mean() * 100 if len(alt) else float("nan")
w(f"  alternative-label matches where the embedding agrees: {agree:.1f}%")
w("")
emb = ad_c[ad_c.method != "exact_preferred"]
w("Embedding similarity of the first candidate (ads without a preferred-label match)")
w(emb["score1"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
w("")
w("Provisional ads per first-candidate ISCO major group (preferred-label match or score >= 0.70)")
ok = (ad_c.method == "exact_preferred") | (ad_c.score1 >= 0.70)
mg = ad_c[ok].occ1_isco.str[0].map(lambda c: f"{c} {isco_label.get(c, '')}").value_counts()
w(mg.to_string())
w("")
w("Provisional ads per TARGET occupation and window, by threshold "
  "(preferred-label matches always counted; final threshold set in Stage 2b)")
rowsT = []
for u, (n, cl) in target_uris.items():
    sel = ad_c.occ1 == u
    r = {"cluster": n, "occupation": occ_label.get(u, "?"), "isco": occ_isco.get(u, "")}
    for th in PROVISIONAL_THRESHOLDS:
        m = sel & ((ad_c.method == "exact_preferred") | (ad_c.score1 >= th))
        r[f"H2>={th:.2f}"] = int((m & (ad_c.window == "H2_2024")).sum())
        r[f"H1>={th:.2f}"] = int((m & (ad_c.window == "H1_2025")).sum())
    rowsT.append(r)
tt = pd.DataFrame(rowsT).sort_values(["cluster", f"H2>={PROVISIONAL_THRESHOLDS[1]:.2f}"],
                                     ascending=[True, False])
tt.to_csv(OUT / "stage2a_target_counts.csv", index=False)
w(tt.to_string(index=False))
w("")
w("Top 40 ESCO occupations among ads (preferred-label match or score >= 0.70), whether target or not")
top_occ = ad_c[ok].groupby("occ1").size().sort_values(ascending=False).head(40)
w(pd.DataFrame({"occupation": top_occ.index.map(occ_label),
                "isco": top_occ.index.map(occ_isco),
                "target": top_occ.index.isin(list(target_uris)),
                "ads": top_occ.values}).to_string(index=False))
w("")
w("Non-target ESCO occupations in industrial ISCO groups with >= 50 ads (candidates to add to the target list)")
ind_groups = ("1321", "1324", "2141", "2144", "2149", "2151", "2152", "3113", "3114", "3115",
              "3118", "3119", "3122", "3139", "3323", "7233", "7412", "7421", "7543")
cnt = ad_c[ok].groupby("occ1").size()
in_ind = np.array([occ_isco.get(u, "") in ind_groups for u in cnt.index], dtype=bool)
extra = cnt[(~cnt.index.isin(list(target_uris))) & in_ind]
extra = extra[extra >= 50].sort_values(ascending=False)
w(pd.DataFrame({"occupation": extra.index.map(occ_label), "isco": extra.index.map(occ_isco),
                "ads": extra.values}).to_string(index=False) if len(extra) else "  none")
w("")
w("30 random embedding matches for a quick eye check (title -> first candidate, score)")
u = emb.drop_duplicates("title_norm")
smp = u.sample(min(30, len(u)), random_state=1)
w(smp[["title_norm", "occ1_label", "score1", "margin", "pref_label", "pref_score"]].round(3).to_string(index=False))
w("")
w(f"Runtime: {time.time() - T0:.0f}s")
(OUT / "stage2a_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
log(f"outputs in {OUT}")
