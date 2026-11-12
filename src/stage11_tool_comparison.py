"""
ZenESCO - Stage 11: quantitative comparison with related taxonomy tools
=======================================================================

Question
    If the decision principle of each related tool is used to edit the same official ESCO profiles from the same
    job ads, which edited taxonomy describes held-out demand best, and which edits do domain experts confirm?

Why re-implementations on common data
    None of the six tools edits occupation-skill links of ESCO for industrial occupations out of the box:
    ESCOPlus adds digital skills from Stack Overflow, ESCOPlus 2.0 extends and prunes software-engineering profiles,
    WETA aligns taxonomies, NEO inserts new occupations, TaxoRef refines the hierarchy and Skills2Graph ranks skills
    for occupations. Running their original code on their original data would compare data sets, not methods.
    Here every tool's published decision rule (parameters as published) is applied to the SAME inputs:
        official graph   ESCO v1.2 profiles of the 16 analysed occupations (Stage 3/6)
        selection data   H2 2024 advertisements, SKILLAB Tracker extraction (Stages 1-2)
        evaluation data  EVAL_WINDOW advertisements (default H1 2025), scored exactly as in Stage 9 / Table 4
    Common rules for every method (also ZenESCO): candidate additions must be visible to the extractor
    (>= 5 ads market-wide, Stage 4) and must not be language or transversal skills (ESCO keeps these outside
    occupation profiles by design). Methods that do not distinguish essential from optional add links as optional.

Adapted decision rules (H2 2024 only)                                                    edit types
    ESCOPlus      (Kavargyris et al., 2026; github.com/dkavargy/ESCOPlus) association rules s -> r between a
                  candidate skill s and an official skill r of o, mined in o's ads; apriori thresholds of the
                  ESCOPlus code: support >= 0.01, confidence >= 0.6, lift > 1  ->  add (o, s)            add
    ESCOPlus 2.0  (Kavargyris et al., 2027; github.com/dkavargy/ESCOPlus2.0) skill intensity SI = n_os / N_o;
                  extension: candidates with SI >= Q0.90 of o's candidates; pruning: official links with
                  SI <= Q0.10 of the SI of the skills seen in o's ads                                 add, delete
    WETA          (Giabelli et al., 2022) embeddings trained on the ads; each candidate skill is aligned to
                  the occupation ranked first by WETA's blend of a cosine-similarity ranking and a neural
                  classification ranking (top n = 5, classifier preferred), trained on the official links   add
    NEO           (Giabelli et al., 2020; 2021) embeddings trained on the ads; adequacy of a skill for an
                  occupation = specificity (similarity to o) minus generality (mean similarity to the other
                  occupations); the 5 most adequate candidates per occupation, adequacy > 0                 add
    TaxoRef       (Malandri et al., 2021; github.com/Crisp-Unimib/TaxoRef) Gaussian naive Bayes on the
                  embeddings of the official links (class = occupation, terms seen >= 20 times, classes
                  with > 2 terms); a link whose predicted occupation differs is moved there               delete, add
    Skills2Graph  (Giabelli et al., 2021) revealed comparative advantage of every skill in every occupation,
                  normalised per occupation (rcaN); required skills rcaN >= 0.6 that are missing are added   add
    ZenESCO (screening)    Stage 6 candidates only (H2 2024 evidence, no held-out test, no review)      all
    ZenESCO (recommended)  Stage 9 recommendations after the expert review (R1_all_links.csv)          all
    Embeddings for WETA / NEO / TaxoRef: concept embeddings of skills and occupations learnt from the
    co-occurrence of concepts in the H2 2024 ads (shifted positive PMI + truncated SVD, 150 dimensions,
    minimum count 5), which is equivalent to skip-gram embeddings (Levy & Goldberg, 2014). The original tools
    train fastText on the ad texts, which the Tracker does not deliver.

Metrics
    held-out (EVAL_WINDOW, identical to Stage 9 / Table 4)
        identification accuracy, MRR (95% bootstrap intervals over ads; paired gain vs original ESCO),
        demand coverage of the top-20 skills, distinctiveness (median 1 - max cosine to another occupation)
    edits      additions, deletions, status changes; essential and protected links removed
    expert     agreement with the expert-verified decisions of review round 1 (365 items, Stage 10):
               precision, recall and F1 of the proposed changes (an edit counts only if its type matches).
               Not computed for ZenESCO (recommended), whose decisions on these items ARE the review.
Caveat
    ZenESCO (recommended) used the H1 2025 ads for its held-out tests and joint guard, so on H1 2025 its scores
    are optimistic; ZenESCO (screening) and the six tools never see H1 2025. For a fully independent test set
    collect a later window (e.g. H2 2025), run Stages 1-2 for it and set EVAL_WINDOW.

Inputs   BASE\\stage1\\ads_clean.parquet, ad_skills_clean.parquet; BASE\\stage2\\ad_occupation.parquet
         BASE\\stage4\\pair_evidence.parquet; BASE\\stage6\\screened_links.parquet, add_candidates.parquet
         BASE\\stage9\\R1_all_links.csv; BASE\\stage10\\review_results.xlsx (or .csv)
         BASE\\esco\\ occupationSkillRelations_en, skills_en, languageSkillsCollection_en, transversalSkillsCollection_en
Outputs  BASE\\stage11\\tool_comparison.csv, tool_edits.csv, F9_tool_comparison.png/pdf, stage11_report.txt
Run      python stage11_tool_comparison.py               # BASE = C:\\skillab_data
         python stage11_tool_comparison.py D:\\data H2_2025   # other BASE and evaluation window
Needs    numpy, pandas, scipy, scikit-learn (WETA classifier), matplotlib, openpyxl (review file)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
EVAL_WINDOW = sys.argv[2] if len(sys.argv) > 2 else "H1_2025"
WIN_SEL = "H2_2024"
OUT = BASE / "stage11"
ESCO = BASE / "esco"
ALPHA = 0.5                 # weight of an optional link (as in Stages 3-9)
TOP_K = 20                  # demand coverage: top-K skills of each occupation's held-out ads
BOOT = 1000                 # bootstrap resamples of held-out ads
SEED = 20261026

# published parameters of the compared tools
ESCOPLUS_SUPPORT, ESCOPLUS_CONF = 0.01, 0.6          # ESCOPlus apriori / association_rules
ESCOPLUS2_Q_ADD, ESCOPLUS2_Q_PRUNE = 0.90, 0.10      # ESCOPlus 2.0 extension (Q90) and pruning (Q10)
ESCOPLUS2_PRUNE_ZERO = True     # True: links never mentioned (SI = 0) are below Q10 and pruned as well;
                                # False: only links mentioned at least once can be flagged (as the GitHub code)
S2G_RCAN = 0.6                  # Skills2Graph: required skills rcaN >= 0.6
NEO_TOP = 5                     # NEO: five suggestions per starting occupation
WETA_TOP = 5                    # WETA: top-5 suggestions blended, first one taken as the automatic alignment
TAXOREF_MIN_COUNT, TAXOREF_MIN_CLASS = 20, 3     # TaxoRef: terms seen >= 20 times, classes with > 2 terms
EMB_DIM, EMB_MIN_COUNT, EMB_CDS = 150, 5, 0.75   # concept embeddings (dimension as in JoTA/WETA)

rng = np.random.default_rng(SEED)
T0 = time.time()
REPORT = []


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def esco_file(name):
    hits = [p for p in ESCO.glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    return hits[0] if hits else None


def read_esco(name, **kw):
    p = esco_file(name)
    return pd.read_csv(p, dtype=str, keep_default_na=False, **kw) if p is not None else None


# ============================================================
# 1. DATA
# ============================================================
log("Loading")
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
ADD6 = pd.read_parquet(BASE / "stage6" / "add_candidates.parquet")
PE = pd.read_parquet(BASE / "stage4" / "pair_evidence.parquet")
OCC = sorted(EV.occ_uri.unique())
OSET = set(OCC)
oi = {o: i for i, o in enumerate(OCC)}
olab = EV.groupby("occ_uri").occupation.first().to_dict()

ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet", columns=["ad_id", "portal", "window", "analysable"])
ads = ads[ads.analysable.astype(bool)].copy()
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
occ_of = occ_of.drop_duplicates("ad_id")
ads = ads.merge(occ_of, on="ad_id", how="left")
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet", columns=["ad_id", "skill_uri"])
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(ads.ad_id))].drop_duplicates()
if EVAL_WINDOW not in set(ads.window):
    sys.exit(f"No analysable ads in window {EVAL_WINDOW}. Windows found: {sorted(ads.window.unique())}")

rel = read_esco("occupationSkillRelations_en.csv", usecols=["occupationUri", "skillUri"])
N_ESCO_OCC = rel.occupationUri.nunique()
gen = rel.groupby("skillUri").occupationUri.nunique()
skills = read_esco("skills_en.csv")
slabel = dict(zip(skills.conceptUri, skills.preferredLabel))
lang, transv = set(), set()
if read_esco("languageSkillsCollection_en.csv") is not None:
    lang = set(read_esco("languageSkillsCollection_en.csv").conceptUri)
if read_esco("transversalSkillsCollection_en.csv") is not None:
    transv = set(read_esco("transversalSkillsCollection_en.csv").conceptUri)
if "reuseLevel" in skills.columns:
    transv |= set(skills.conceptUri[skills.reuseLevel == "transversal"])

say("ZenESCO - Stage 11 report: quantitative comparison with related taxonomy tools")
say("=" * 78)
say(f"Selection window {WIN_SEL}, evaluation window {EVAL_WINDOW}; {len(OCC)} occupations, "
    f"{len(EV):,} official links")

# ---- original profiles ----
P0 = {o: {} for o in OCC}
for o, s, r in zip(EV.occ_uri, EV.skill_uri, EV.relation):
    P0[o][s] = 1.0 if r == "essential" else ALPHA
protected = set(zip(EV.occ_uri[EV.protected_any.astype(bool)], EV.skill_uri[EV.protected_any.astype(bool)]))


def copy_prof(P):
    return {o: dict(p) for o, p in P.items()}


# ---- selection-window evidence (Stage 4, H2 2024) ----
pe = PE[PE.occ_uri.isin(OSET)].copy()
pe["relation"] = pe.relation.fillna("").astype(str)
pe["si"] = pe.n / pe.N.clip(lower=1)
pe["in_scope"] = pe.visible.astype(bool) & ~pe.skill_uri.isin(lang) & ~pe.skill_uri.isin(transv)
POOL = pe[(pe.relation == "") & (pe.n >= 1) & pe.in_scope]          # common candidate pool for additions
pool_of = POOL.groupby("occ_uri").skill_uri.agg(set).to_dict()
say(f"Common candidate pool for additions: {len(POOL):,} (occupation, skill) pairs seen in {WIN_SEL} ads "
    f"(visible, not language/transversal, not yet linked)")

sel = ads[ads.window == WIN_SEL].reset_index(drop=True)
sk_sel = ask[ask.ad_id.isin(set(sel.ad_id))]


# ============================================================
# 2. METHODS (each returns an edited copy of P0)
# ============================================================
METHODS, NOTES = {}, {}


def add_link(P, o, s, w=ALPHA):
    if s not in P[o]:
        P[o][s] = w


# ---- ESCOPlus: association rules candidate -> official skill, within each occupation's ads ----
log("ESCOPlus")
P = copy_prof(P0)
n_rules = 0
for o in OCC:
    a_o = sel.ad_id[sel.occ_uri == o]
    if len(a_o) == 0 or o not in pool_of:
        continue
    s_o = sk_sel[sk_sel.ad_id.isin(set(a_o))]
    ai = {a: i for i, a in enumerate(a_o)}
    sk_list = sorted(s_o.skill_uri.unique())
    kk = {s: i for i, s in enumerate(sk_list)}
    Y = sparse.csr_matrix((np.ones(len(s_o)), (s_o.ad_id.map(ai).values, s_o.skill_uri.map(kk).values)),
                          shape=(len(a_o), len(sk_list)))
    Y.data[:] = 1.0
    cand = [s for s in sk_list if s in pool_of[o]]
    offi = [s for s in sk_list if s in P0[o]]
    if not cand or not offi:
        continue
    Yc, Yr = Y[:, [kk[s] for s in cand]], Y[:, [kk[s] for s in offi]]
    co = (Yc.T @ Yr).toarray()                                   # ads with both s and r
    nc = np.asarray(Yc.sum(axis=0)).ravel()
    nr = np.asarray(Yr.sum(axis=0)).ravel()
    N = float(len(a_o))
    supp = co / N
    conf = co / np.maximum(nc[:, None], 1)
    lift = conf / np.maximum(nr[None, :] / N, 1e-12)
    ok = (supp >= ESCOPLUS_SUPPORT) & (conf >= ESCOPLUS_CONF) & (lift > 1)
    n_rules += int(ok.sum())
    for j in np.where(ok.any(axis=1))[0]:
        add_link(P, o, cand[j])
METHODS["ESCOPlus"] = P
NOTES["ESCOPlus"] = f"{n_rules} rules candidate -> official skill (support >= {ESCOPLUS_SUPPORT}, confidence >= {ESCOPLUS_CONF}, lift > 1)"

# ---- ESCOPlus 2.0: skill intensity, Q90 extension and Q10 pruning ----
log("ESCOPlus 2.0")
P = copy_prof(P0)
for o in OCC:
    po = pe[pe.occ_uri == o]
    c = po[(po.relation == "") & (po.n >= 1) & po.in_scope]
    if len(c):
        q90 = np.quantile(c.si, ESCOPLUS2_Q_ADD)
        for s in c.skill_uri[c.si >= q90]:
            add_link(P, o, s)
    seen = po[po.n >= 1]
    if len(seen):
        q10 = np.quantile(seen.si, ESCOPLUS2_Q_PRUNE)
        off = po[po.relation != ""]
        flag = off[(off.si <= q10) & ((off.n >= 1) | ESCOPLUS2_PRUNE_ZERO)]
        for s in flag.skill_uri:
            P[o].pop(s, None)
METHODS["ESCOPlus 2.0"] = P
NOTES["ESCOPlus 2.0"] = (f"extension SI >= Q{int(ESCOPLUS2_Q_ADD * 100)} of candidates; pruning SI <= "
                         f"Q{int(ESCOPLUS2_Q_PRUNE * 100)} of skills seen (links never mentioned "
                         f"{'included' if ESCOPLUS2_PRUNE_ZERO else 'excluded'})")

# ---- Skills2Graph: revealed comparative advantage, normalised per occupation ----
log("Skills2Graph")
occ_all = sel.occ_uri.fillna("_unassigned").values
ou = {o: i for i, o in enumerate(sorted(set(occ_all)))}
vis = set(pe.skill_uri[pe.visible.astype(bool)]) | set(
    sk_sel.groupby("skill_uri").ad_id.nunique().loc[lambda x: x >= 5].index)
sv = sk_sel[sk_sel.skill_uri.isin(vis)].merge(sel[["ad_id"]].assign(oo=occ_all), on="ad_id")
vk = sorted(vis)
vki = {s: i for i, s in enumerate(vk)}
cnt = sv.groupby(["oo", "skill_uri"]).ad_id.nunique().reset_index()
n_ads_o = pd.Series(occ_all).value_counts()
SF = sparse.csr_matrix((cnt.ad_id.values / cnt.oo.map(n_ads_o).values,
                        (cnt.oo.map(ou).values, cnt.skill_uri.map(vki).values)), shape=(len(ou), len(vk))).toarray()
share_o = SF / np.maximum(SF.sum(axis=1, keepdims=True), 1e-12)
share_m = SF.sum(axis=0) / max(SF.sum(), 1e-12)
RCA = share_o / np.maximum(share_m[None, :], 1e-12)
RCAN = RCA / np.maximum(RCA.max(axis=1, keepdims=True), 1e-12)
P = copy_prof(P0)
for o in OCC:
    if o not in ou:
        continue
    req = {vk[j] for j in np.where(RCAN[ou[o]] >= S2G_RCAN)[0]}
    for s in req & pool_of.get(o, set()):
        add_link(P, o, s)
METHODS["Skills2Graph"] = P
NOTES["Skills2Graph"] = f"rcaN >= {S2G_RCAN} over {len(ou)} occupation groups of the {WIN_SEL} ads"

# ---- concept embeddings from the H2 2024 ads (for WETA, NEO, TaxoRef) ----
log("Concept embeddings (SPPMI + SVD)")
tok = pd.concat([sk_sel[["ad_id", "skill_uri"]].rename(columns={"skill_uri": "t"}),
                 sel.loc[sel.occ_uri.notna(), ["ad_id", "occ_uri"]].rename(columns={"occ_uri": "t"})
                 .assign(t=lambda d: "OCC::" + d.t.astype(str))]).drop_duplicates()
tc = tok.groupby("t").ad_id.nunique()
vocab = sorted(tc.index[tc >= EMB_MIN_COUNT])
vi = {t: i for i, t in enumerate(vocab)}
tok = tok[tok.t.isin(vi)]
adi = {a: i for i, a in enumerate(sorted(tok.ad_id.unique()))}
Bm = sparse.csr_matrix((np.ones(len(tok)), (tok.ad_id.map(adi).values, tok.t.map(vi).values)),
                       shape=(len(adi), len(vocab)))
Bm.data[:] = 1.0
M = (Bm.T @ Bm).tocoo()
keep = M.row != M.col
r_, c_, m_ = M.row[keep], M.col[keep], M.data[keep].astype(float)
tot = m_.sum()
rs = np.bincount(r_, weights=m_, minlength=len(vocab))
cs = np.bincount(c_, weights=m_, minlength=len(vocab)) ** EMB_CDS
pmi = np.log(m_ * tot / np.maximum(rs[r_], 1e-12)) - np.log(cs[c_] / cs.sum() * tot)
pos = pmi > 0
PPMI = sparse.csr_matrix((pmi[pos], (r_[pos], c_[pos])), shape=(len(vocab), len(vocab)))
k_ = int(min(EMB_DIM, len(vocab) - 2))
U, Sv, _ = svds(PPMI.astype(float), k=k_, v0=np.full(len(vocab), 1.0 / np.sqrt(len(vocab))))   # deterministic
EMB = U * np.sqrt(Sv)[None, :]
EMB = EMB / np.maximum(np.linalg.norm(EMB, axis=1, keepdims=True), 1e-12)


def emb(t):
    return EMB[vi[t]] if t in vi else None


occ_tok = {o: "OCC::" + o for o in OCC if ("OCC::" + o) in vi}
OE = np.array([EMB[vi[occ_tok[o]]] for o in OCC if o in occ_tok])
OE_occ = [o for o in OCC if o in occ_tok]
say(f"Concept embeddings: {len(vocab):,} concepts (>= {EMB_MIN_COUNT} ads), {k_} dimensions; "
    f"{len(OE_occ)} of {len(OCC)} occupations have an embedding")

# ---- NEO: adequacy = specificity - generality, top 5 per occupation ----
log("NEO")
P = copy_prof(P0)
for j, o in enumerate(OE_occ):
    c = [s for s in sorted(pool_of.get(o, set())) if s in vi]
    if not c:
        continue
    sims = EMB[[vi[s] for s in c]] @ OE.T                     # candidates x occupations
    spec = sims[:, j]
    genr = (sims.sum(axis=1) - spec) / max(len(OE_occ) - 1, 1)
    adeq = spec - genr
    order = np.argsort(-adeq)[:NEO_TOP]
    for i in order:
        if adeq[i] > 0:
            add_link(P, o, c[i])
METHODS["NEO"] = P
NOTES["NEO"] = f"top {NEO_TOP} candidates per occupation by adequacy (specificity - generality) > 0"

# ---- WETA: blended cosine + classification ranking, first suggestion ----
log("WETA")
P = copy_prof(P0)
tr = [(s, o) for o in OE_occ for s in P0[o] if s in vi]
cand_sk = sorted({s for o in OCC for s in pool_of.get(o, set()) if s in vi})
n_weta = 0
try:
    from sklearn.neural_network import MLPClassifier
    Xtr = np.array([EMB[vi[s]] for s, _ in tr])
    ytr = np.array([OE_occ.index(o) for _, o in tr])
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=500,
                        random_state=SEED)
    clf.fit(Xtr, ytr)
    Xc = np.array([EMB[vi[s]] for s in cand_sk])
    prob = np.zeros((len(cand_sk), len(OE_occ)))
    prob[:, clf.classes_] = clf.predict_proba(Xc)
    cosv = Xc @ OE.T
    for i, s in enumerate(cand_sk):
        top_cos = list(np.argsort(-cosv[i])[:WETA_TOP])
        top_cls = list(np.argsort(-prob[i])[:WETA_TOP])
        agree = [k for k in top_cls if k in top_cos]           # concepts both methods suggest
        blend = agree + [k for k in top_cls if k not in agree] + [k for k in top_cos if k not in top_cls]
        o = OE_occ[blend[0]]
        if s in pool_of.get(o, set()):
            add_link(P, o, s)
            n_weta += 1
    NOTES["WETA"] = (f"{len(cand_sk)} candidate skills aligned (classifier trained on {len(tr)} official links); "
                     f"{n_weta} aligned to an occupation in whose ads they occur")
except ImportError:
    NOTES["WETA"] = "scikit-learn not installed - WETA skipped"
    P = None
if P is not None:
    METHODS["WETA"] = P

# ---- TaxoRef: Gaussian naive Bayes on link embeddings, move misclassified links ----
log("TaxoRef")
P = copy_prof(P0)
rows = [(o, s) for o in OE_occ for s in P0[o] if s in vi and tc.get(s, 0) >= TAXOREF_MIN_COUNT]
cls_n = pd.Series([o for o, _ in rows]).value_counts()
rows = [(o, s) for o, s in rows if cls_n.get(o, 0) >= TAXOREF_MIN_CLASS]
n_moved = 0
if rows:
    Xt = np.array([EMB[vi[s]] for _, s in rows])
    cl = sorted({o for o, _ in rows})
    yt = np.array([cl.index(o) for o, _ in rows])
    mu = np.array([Xt[yt == k].mean(axis=0) for k in range(len(cl))])
    var = np.array([Xt[yt == k].var(axis=0) for k in range(len(cl))]) + 1e-9 * Xt.var(axis=0).max()
    prior = np.log(np.bincount(yt) / len(yt))
    ll = (-0.5 * (np.log(2 * np.pi * var)[None, :, :] + (Xt[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :])
          ).sum(axis=2) + prior[None, :]
    pred = ll.argmax(axis=1)
    for (o, s), k in zip(rows, pred):
        o2 = cl[k]
        if o2 != o:
            w = P0[o][s]
            P[o].pop(s, None)
            add_link(P, o2, s, w)
            n_moved += 1
METHODS["TaxoRef"] = P
NOTES["TaxoRef"] = f"{len(rows)} official links classified, {n_moved} moved to the predicted occupation"

# ---- ZenESCO (screening): Stage 6 candidates, H2 2024 only ----
P = copy_prof(P0)
for o, s, a in zip(EV.occ_uri, EV.skill_uri, EV.action):
    if a == "DELETE candidate":
        P[o].pop(s, None)
    elif a == "DEMOTE candidate":
        P[o][s] = ALPHA
    elif a == "UPGRADE candidate":
        P[o][s] = 1.0
for o, s, t in zip(ADD6.occ_uri, ADD6.skill_uri, ADD6.proposed_type):
    if o in OSET:
        add_link(P, o, s, 1.0 if t == "essential" else ALPHA)
METHODS["ZenESCO (screening)"] = P
NOTES["ZenESCO (screening)"] = "Stage 6 delete/demote/upgrade/add candidates (H2 2024 evidence only)"

# ---- ZenESCO (recommended): Stage 9 after expert review ----
r1f = BASE / "stage9" / "R1_all_links.csv"
if r1f.exists():
    R1 = pd.read_csv(r1f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    P = copy_prof(P0)
    for o, s, rec in zip(R1.occ_uri, R1.skill_uri, R1.recommendation):
        if o not in OSET:
            continue
        if rec == "Delete":
            P[o].pop(s, None)
        elif rec in ("Upgrade -> optional", "Add (optional)"):
            P[o][s] = ALPHA
        elif rec in ("Upgrade -> essential", "Add (essential)"):
            P[o][s] = 1.0
    METHODS["ZenESCO (recommended)"] = P
    NOTES["ZenESCO (recommended)"] = "Stage 9 recommendations after the expert review (R1_all_links.csv)"
else:
    say("  R1_all_links.csv not found - run Stage 9 first; ZenESCO (recommended) skipped")

ORDER = ["Original ESCO", "ESCOPlus", "ESCOPlus 2.0", "WETA", "NEO", "TaxoRef", "Skills2Graph",
         "ZenESCO (screening)", "ZenESCO (recommended)"]
METHODS = dict([("Original ESCO", P0)] + [(k, METHODS[k]) for k in ORDER[1:] if k in METHODS])

# ============================================================
# 3. HELD-OUT EVALUATION (same scoring as Stage 9 / Table 4)
# ============================================================
log(f"Held-out evaluation on {EVAL_WINDOW}")
test = ads[(ads.window == EVAL_WINDOW) & ads.occ_uri.isin(OSET)].reset_index(drop=True)
at = ask[ask.ad_id.isin(set(test.ad_id))]
SK = sorted(set(at.skill_uri) | {s for P in METHODS.values() for p in P.values() for s in p})
si = {s: i for i, s in enumerate(SK)}
aid = {a: i for i, a in enumerate(test.ad_id)}
X = sparse.csr_matrix((np.ones(len(at)), (at.ad_id.map(aid).values, at.skill_uri.map(si).values)),
                      shape=(len(test), len(SK)))
X.data[:] = 1.0
idf = np.maximum(np.array([np.log(N_ESCO_OCC / gen.get(s, 1)) for s in SK]), 0.05)
Xn = (sparse.diags(1.0 / np.maximum(np.sqrt(np.asarray(X.multiply(X).multiply(idf ** 2).sum(axis=1)).ravel()), 1e-12))
      @ X.multiply(idf).tocsr()).tocsr()
true = test.occ_uri.map(oi).values
h1prev = (at.merge(test[["ad_id", "occ_uri"]], on="ad_id").groupby(["occ_uri", "skill_uri"]).ad_id.nunique()
          / test.groupby("occ_uri").size()).rename("p").reset_index()
TOPS = {o: h1prev[(h1prev.occ_uri == o) & ~h1prev.skill_uri.isin(lang)].nlargest(TOP_K, "p").skill_uri.tolist()
        for o in OCC}
BIDX = rng.integers(0, len(test), size=(BOOT, len(test)))


def score(P):
    W = np.zeros((len(OCC), len(SK)))
    for o, p in P.items():
        for s, w in p.items():
            W[oi[o], si[s]] = w
    V = W * idf
    Vn = V / np.maximum(np.sqrt((V ** 2).sum(axis=1)), 1e-12)[:, None]
    S = np.asarray(Xn @ Vn.T)
    ts = S[np.arange(len(S)), true]
    acc = (((S > ts[:, None]).sum(axis=1) == 0) & ((S == ts[:, None]).sum(axis=1) == 1) & (ts > 0)).astype(float)
    rr = 1.0 / (1 + (S > ts[:, None]).sum(axis=1))
    C = Vn @ Vn.T
    np.fill_diagonal(C, -1)
    cov = np.nanmean([np.mean([s in P[o] for s in TOPS[o]]) if TOPS[o] else np.nan for o in OCC])
    return acc, rr, float(np.median(1 - C.max(axis=1))), float(cov)


def edits(P):
    d = []
    for o in OCC:
        for s, w in P[o].items():
            w0 = P0[o].get(s)
            if w0 is None:
                d.append((o, s, "add (essential)" if w == 1.0 else "add (optional)"))
            elif w != w0:
                d.append((o, s, "upgrade -> essential" if w > w0 else "upgrade -> optional"))
        for s in P0[o]:
            if s not in P[o]:
                d.append((o, s, "delete"))
    return pd.DataFrame(d, columns=["occ_uri", "skill_uri", "edit"])


# ---- expert-verified reference: review round 1 (Stage 10) ----
def read_review():
    for f in [BASE / "stage10" / "review_results.xlsx", BASE / "stage10" / "review_results.csv"]:
        if not f.exists():
            continue
        if f.suffix == ".csv":
            return pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=None, engine="python")
        try:
            return pd.read_excel(f, sheet_name="Review", dtype=str, engine="openpyxl").fillna("")
        except Exception:
            import openpyxl
            ws = openpyxl.load_workbook(f, read_only=True, data_only=True)["Review"]
            rr_ = [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
            return pd.DataFrame(rr_[1:], columns=rr_[0])
    return None


CLASS = {"Retain": "keep", "Reject addition": "keep", "Delete": "delete", "Upgrade -> optional": "upgrade -> optional",
         "Upgrade -> essential": "upgrade -> essential", "Add (essential)": "add", "Add (optional)": "add"}
REF = None
rv = read_review()
if rv is not None:
    rv.columns = [str(c).strip() for c in rv.columns]
    if "Round" in rv.columns:                       # combined file (rounds 1 and 2): keep round 1
        rv = rv[rv["Round"].astype(str).str.strip().isin(["1", "1.0"])].copy()
    elif "Item type" in rv.columns:                 # round-1 file: audit + expert review items (no "check" items)
        rv = rv[rv["Item type"].astype(str).str.strip() != "check"].copy()
    for c in ["user_agree", "user_recommendation", "Final recommendation", "ESCO now", "occ_uri", "skill_uri"]:
        if c not in rv.columns:
            rv[c] = ""
    rv = rv.fillna("").astype(str)
    agree = rv["user_agree"].str.strip().str.upper()
    final = np.where((agree == "N") & (rv["user_recommendation"].str.strip() != ""),
                     rv["user_recommendation"].str.strip(), rv["Final recommendation"].str.strip())
    cand = rv["ESCO now"].str.strip() == "not in ESCO"
    final = np.where(cand & (final == "Delete"), "Reject addition", final)
    final = np.where(~cand & (final == "Reject addition"), "Delete", final)
    REF = pd.DataFrame({"occ_uri": rv.occ_uri.values, "skill_uri": rv.skill_uri.values,
                        "ref": pd.Series(final).map(CLASS).values})
    REF = REF[REF.ref.notna() & REF.occ_uri.isin(OSET)].drop_duplicates(["occ_uri", "skill_uri"])
    say(f"Expert-verified reference: {len(REF)} round-1 review items "
        f"({(REF.ref != 'keep').sum()} changes confirmed, {(REF.ref == 'keep').sum()} kept or rejected)")
else:
    say("  review_results.xlsx not found - expert agreement not computed")


def expert_agreement(P):
    if REF is None:
        return {}
    pred = []
    for o, s in zip(REF.occ_uri, REF.skill_uri):
        w0, w = P0[o].get(s), P[o].get(s)
        if w0 is None:
            pred.append("add" if w is not None else "keep")
        elif w is None:
            pred.append("delete")
        elif w != w0:
            pred.append("upgrade -> essential" if w > w0 else "upgrade -> optional")
        else:
            pred.append("keep")
    pred = np.array(pred)
    ref = REF.ref.values
    tp = ((pred == ref) & (ref != "keep")).sum()
    npred, ntrue = (pred != "keep").sum(), (ref != "keep").sum()
    prec = tp / npred if npred else np.nan
    rec = tp / ntrue if ntrue else np.nan
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return {"expert_items": len(ref), "expert_proposed": int(npred), "expert_precision": prec,
            "expert_recall": rec, "expert_f1": f1, "expert_accuracy": float((pred == ref).mean())}


rows, ALL_EDITS = [], []
base_acc, base_rr, _, _ = score(P0)
for name, P in METHODS.items():
    acc, rr, dist, cov = score(P)
    ba = acc[BIDX].mean(axis=1)
    bm = rr[BIDX].mean(axis=1)
    gain = ba - base_acc[BIDX].mean(axis=1)
    E = edits(P)
    E.insert(0, "method", name)
    ALL_EDITS.append(E)
    ess_del = sum(1 for o, s, e in zip(E.occ_uri, E.skill_uri, E.edit) if e == "delete" and P0[o][s] == 1.0)
    prot = sum(1 for o, s in zip(E.occ_uri, E.skill_uri) if (o, s) in protected)
    row = {"method": name, "acc": acc.mean(), "acc_lo": np.percentile(ba, 2.5), "acc_hi": np.percentile(ba, 97.5),
           "gain_lo": np.percentile(gain, 2.5), "gain_hi": np.percentile(gain, 97.5),
           "mrr": rr.mean(), "mrr_lo": np.percentile(bm, 2.5), "mrr_hi": np.percentile(bm, 97.5),
           "coverage_top20": cov, "distinctiveness": dist,
           "links_after": sum(len(p) for p in P.values()),
           "n_add": int(E.edit.str.startswith("add").sum()), "n_delete": int((E.edit == "delete").sum()),
           "n_status": int(E.edit.str.startswith("upgrade").sum()),
           "essential_deleted": ess_del, "protected_changed": prot, "note": NOTES.get(name, "")}
    if name != "ZenESCO (recommended)":
        row.update(expert_agreement(P))
    rows.append(row)
RES = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
RES.to_csv(OUT / "tool_comparison.csv", index=False, encoding="utf-8-sig")
ED = pd.concat(ALL_EDITS, ignore_index=True)
ED["occupation"] = ED.occ_uri.map(olab)
ED["skill"] = ED.skill_uri.map(slabel)
ED.to_csv(OUT / "tool_edits.csv", index=False, encoding="utf-8-sig")

say("")
say(f"Held-out results ({EVAL_WINDOW}, {len(test):,} ads; chance accuracy {1 / len(OCC):.3f}). "
    f"Gain = paired 95% bootstrap interval of the accuracy change vs original ESCO.")
hdr = (f"{'method':24s} {'acc':>6s} {'gain 95% CI':>18s} {'MRR':>6s} {'cov20':>6s} {'dist':>6s} "
       f"{'+add':>5s} {'-del':>5s} {'stat':>5s} {'essDel':>6s} {'prot':>5s} {'P':>5s} {'R':>5s} {'F1':>5s}")
say(hdr)
say("-" * len(hdr))
for r in RES.to_dict("records"):
    f = lambda k: f"{r[k]:.2f}" if k in r and pd.notna(r.get(k)) else "  n/a"
    say(f"{r['method']:24s} {r['acc']:6.3f} [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}] {r['mrr']:6.3f} "
        f"{r['coverage_top20']:6.3f} {r['distinctiveness']:6.3f} {r['n_add']:5d} {r['n_delete']:5d} "
        f"{r['n_status']:5d} {r['essential_deleted']:6d} {r['protected_changed']:5d} "
        f"{f('expert_precision'):>5s} {f('expert_recall'):>5s} {f('expert_f1'):>5s}")
say("")
say("Notes per method")
for k in METHODS:
    if k in NOTES:
        say(f"  {k}: {NOTES[k]}")
say("  Expert P/R/F1: precision, recall and F1 of the proposed changes against the expert-verified round-1 "
    "decisions; n/a for ZenESCO (recommended), whose decisions on these items are the review itself.")
if EVAL_WINDOW == "H1_2025":
    say("  ZenESCO (recommended) used H1 2025 for its held-out tests and joint guard: its scores on H1 2025 are "
        "optimistic. ZenESCO (screening) and the six tools used H2 2024 only.")

# ============================================================
# 4. FIGURE F9 (bars in the colours of Fig. 6 / Stage 7)
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    COL = {"Original ESCO": "#c9c8c2", "ESCOPlus": "#eb6834", "ESCOPlus 2.0": "#eda100", "WETA": "#2a78d6",
           "NEO": "#1baf7a", "TaxoRef": "#e87ba4", "Skills2Graph": "#6250d6",
           "ZenESCO (screening)": "#f0a09f", "ZenESCO (recommended)": "#e34948"}
    SHORT = {"Original ESCO": "Original\nESCO", "ESCOPlus 2.0": "ESCOPlus\n2.0", "Skills2Graph": "Skills2\nGraph",
             "ZenESCO (screening)": "ZenESCO\nscreening", "ZenESCO (recommended)": "ZenESCO\nrecomm."}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
    names = RES.method.tolist()
    x = np.arange(len(names))
    cols = [COL.get(n, MUTED) for n in names]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.3))
    panels = [("acc", "Identification accuracy", "(a) Identification", True),
              ("mrr", "Mean reciprocal rank", "(b) Ranking", True),
              ("coverage_top20", f"Coverage of top-{TOP_K} skills", "(c) Demand coverage", False),
              ("distinctiveness", "Distinctiveness", "(d) Distinctiveness", False),
              (None, "Number of edits", "(e) Edits proposed", False),
              ("expert_f1", "F1 vs expert decisions", "(f) Agreement with experts", False)]
    for ax, (met, ylab, cap, ci) in zip(axs.ravel(), panels):
        if met is None:
            a_, d_, s_ = RES.n_add.values, RES.n_delete.values, RES.n_status.values
            ax.bar(x, a_, color=cols, width=0.72, zorder=3)
            ax.bar(x, d_, bottom=a_, color=cols, width=0.72, alpha=0.55, hatch="////", ec="white", lw=0, zorder=3)
            ax.bar(x, s_, bottom=a_ + d_, color=cols, width=0.72, alpha=0.25, zorder=3)
            for i, t in enumerate(a_ + d_ + s_):
                ax.text(i, t, f"{t:,}", ha="center", va="bottom", fontsize=5.6, color=INK2)
            ax.legend(handles=[Patch(fc=MUTED, label="additions"),
                               Patch(fc=MUTED, alpha=0.55, hatch="////", ec="white", label="deletions"),
                               Patch(fc=MUTED, alpha=0.25, label="status changes")],
                      fontsize=5.8, loc="upper left")
            if (a_ + d_ + s_).max() > 20 * max(np.median(a_ + d_ + s_), 1):
                ax.set_yscale("symlog", linthresh=10)
        else:
            v = RES[met].values.astype(float) if met in RES else np.full(len(names), np.nan)
            err = None
            if ci:
                err = np.vstack([v - RES[met + "_lo"].values, RES[met + "_hi"].values - v])
            ax.bar(x, np.nan_to_num(v), color=cols, width=0.72, yerr=err, error_kw=dict(lw=0.7, capsize=1.6,
                   ecolor=INK2), zorder=3)
            for i, t in enumerate(v):
                ax.text(i, (0 if np.isnan(t) else t) + (0 if err is None else err[1][i]),
                        "n/a" if np.isnan(t) else f"{t:.3f}" if t < 1 else f"{t:.2f}",
                        ha="center", va="bottom", fontsize=5.4, color=INK2, rotation=90 if not np.isnan(t) else 0)
            if met != "expert_f1":
                ax.axhline(RES[met].iloc[0], color=INK2, lw=0.5, ls=(0, (1, 2)), zorder=4)
            lo_ = np.nanmin(v if err is None else v - err[0])
            hi_ = np.nanmax(v if err is None else v + err[1])
            if met == "distinctiveness":
                ax.set_ylim(max(0, lo_ - 0.12), min(1, hi_ + 0.08))
            else:
                ax.set_ylim(0, min(hi_ * 1.25, 1.12) if hi_ > 0 else 1)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.1, cap, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK)
    fig.legend(handles=[Patch(fc=c, label=n) for n, c in zip(names, cols)], loc="upper center", ncol=5,
               fontsize=6.8, bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(h_pad=2.6, w_pad=1.2)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"F9_tool_comparison.{ext}", dpi=300, bbox_inches="tight")
    log("F9 saved")
except Exception as ex:
    import traceback
    traceback.print_exc()
    log(f"F9 skipped: {ex!r}")

say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "stage11_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
