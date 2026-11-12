"""
ZenESCO - Stage 9: final taxonomy adjustment (Retain / Delete / Upgrade / Add / Expert review)
==============================================================================================

Every official ESCO link of the analysed occupations, and every addition candidate, gets exactly ONE
recommendation with its full evidence trail, following the decision rules summarised in README.md.

Evidence used (all computed earlier, except E6 which is computed here)
    E1 demand        share of the occupation's ads mentioning the skill (H2 2024; H1 2025 as a check)
    E2 specificity   Pr(skill over-represented in the occupation's ads), lift
    E3 observability visible / powered after the advertising discount; implied by a related skill (Stage 5)
    E4 plausibility  semantic percentile (if computed in Stage 4)
    E5 held-out test change in identification on H1 2025 ads (Stage 7): non-inferior / clear gain
    E6 stability     share of 200 bootstrap resamples of the H2 2024 ads in which the evidence rule still
                     holds (selection probability), plus same direction in H1 2025

Decision rules (first match wins)
  Existing ESCO link
    protected (safety/legal, transversal, last link)              -> Retain            reason: protected
    absence not informative                                         -> Retain            reason: no evidence
    implied by a supported related skill                            -> Retain            reason: implied
    weak, optional, non-inferior on held-out ads, stability >= 0.8 -> Delete
    weak, essential, same conditions                                -> Upgrade -> optional
    optional, demanded like essential links, non-inferior, stab >= 0.8 -> Upgrade -> essential
    any of these, but held-out test fails or stability 0.5-0.8      -> Expert review     reason: conflict-...
    otherwise                                                       -> Retain            reason: supported / adequate
  Pair not in ESCO (addition candidate from Stage 6)
    low semantic plausibility (E4 < 25th percentile)                -> Expert review     reason: low-plausibility
    clear gain on held-out ads (90% interval > 0), stab >= 0.8,
      same direction in H1 2025                                     -> Add (essential / optional)
    harms held-out identification (dU below the margin m_U)         -> Reject addition
    otherwise (demand and specificity pass, the rest does not)      -> Expert review
  Confidence: Firm = stability >= 0.9 and same direction in H1 2025; Tentative = otherwise.
  Expert review (Stage 10): if BASE\\stage10\\review_results.xlsx exists, its final decisions are applied after
  the joint guard (the validator's decision where user_agree = N); changed items get confidence "Expert", R1 gets
  a column expert_review (resolved / confirmed / overruled), and the report gives how many decisions the expert changed.

Outputs (BASE\\stage9\\)
    R1_all_links.csv             every link and candidate: recommendation, confidence, reason, evidence
    R1_main_text.csv             the strongest changes per occupation (2 per edit type) for the paper
    R2_occupation_summary.csv    per occupation: counts, coverage and identification before -> after
    updated_profiles.csv         the updated ESCO profile of every analysed occupation
    ZenESCO_tables.xlsx          all of the above as sheets (supplement)
    F8_alignment_before_after.png/pdf, stage9_report.txt
Run   python stage9_final_adjustment.py            # BASE = C:\\skillab_data
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
OUT = BASE / "stage9"
ESCO = BASE / "esco"
ALPHA = 0.5
CRED = 0.95
PRIOR_STRENGTH = 2.0          # same prior as Stage 4
B = 200                        # bootstrap resamples for stability
STAB_OK, STAB_FIRM, STAB_LOW = 0.8, 0.9, 0.5
SEM_LOW = 25
TOP_K = 20
STRUCT_PCT = 50                # E4a structural plausibility: closeness(o, s) = similarity of o to the most
                               # similar ESCO occupation that uses s (IDF-cosine of ESCO profiles). An addition
                               # must be closer than the median RANDOM ESCO skill (50th percentile of closeness
                               # over all ESCO skills not in o's profile)
SEM_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # E4b semantic plausibility (rule A3): cosine of skill and
                               # occupation texts, as a percentile of o's own ESCO links; flag below SEM_LOW
FOUND_KNOW, FOUND_ANY = 30, 110  # absence is not informative for knowledge used by >= 30 ESCO occupations
                               # or any skill used by >= 110 (foundational / general; Doc 1 section 3.3)
MAX_GUARD = 60                 # joint guard: maximum number of changes it may send back to expert review
SEED = 20261026
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


# ============================================================
# 1. LOAD
# ============================================================
log("Loading")
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
ADD = pd.read_parquet(BASE / "stage6" / "add_candidates.parquet")
EF = pd.read_parquet(BASE / "stage7" / "edit_effects.parquet")
m_U = float(pd.read_csv(BASE / "stage7" / "baseline.csv").m_U.iloc[0])
OCC = sorted(EV.occ_uri.unique())
olab = EV.groupby("occ_uri").occupation.first().to_dict()
ocl = EV.groupby("occ_uri").cluster.first().to_dict()

ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet", columns=["ad_id", "portal", "window", "analysable"])
ads = ads[ads.analysable & ads.window.isin(["H2_2024", "H1_2025"])].copy()
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
ads = ads.merge(occ_of, on="ad_id")
ads = ads[ads.occ_uri.isin(set(OCC))].reset_index(drop=True)
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(ads.ad_id))].drop_duplicates()

rel = pd.read_csv(esco_file("occupationSkillRelations_en.csv"), dtype=str,
                  usecols=["occupationUri", "skillUri", "relationType"])
N_OCC = rel.occupationUri.nunique()
gen = rel.groupby("skillUri").occupationUri.nunique()
# ESCO neighbourhood of each analysed occupation (for structural plausibility of additions)
_oi = {o: i for i, o in enumerate(sorted(rel.occupationUri.unique()))}
_si = {s_: i for i, s_ in enumerate(sorted(rel.skillUri.unique()))}
_w = np.where(rel.relationType == "essential", 1.0, 0.5) * np.log(N_OCC / rel.skillUri.map(gen).values)
_W = sparse.csr_matrix((_w, (rel.occupationUri.map(_oi).values, rel.skillUri.map(_si).values)),
                       shape=(len(_oi), len(_si)))
_n = np.sqrt(np.asarray(_W.multiply(_W).sum(axis=1)).ravel())
_Wn = sparse.diags(1.0 / np.maximum(_n, 1e-12)) @ _W
_inv = {i: o for o, i in _oi.items()}
used_by = rel.groupby("skillUri").occupationUri.agg(set).to_dict()
occs = pd.read_csv(esco_file("occupations_en.csv"), dtype=str, usecols=["conceptUri", "iscoGroup"])
isco = dict(zip(occs.conceptUri, occs.iscoGroup))
skills = pd.read_csv(esco_file("skills_en.csv"), dtype=str, keep_default_na=False, usecols=["conceptUri", "preferredLabel"])
slabel = dict(zip(skills.conceptUri, skills.preferredLabel))
lang = set()
p_ = esco_file("languageSkillsCollection_en.csv")
if p_ is not None:
    lang = set(pd.read_csv(p_, dtype=str).conceptUri)

CLOSE, CNULL = {}, {}
_B = (_W > 0).astype(float).tocsc()
_isk = np.array(sorted(_si, key=_si.get))
for o in OCC:
    if o in _oi:
        sims = np.asarray((_Wn[_oi[o]] @ _Wn.T).todense()).ravel()
        sims[_oi[o]] = -1
        cl = np.asarray(_B.multiply(sims[:, None]).tocsc().max(axis=0).todense()).ravel()
        CLOSE[o] = dict(zip(_isk, cl))
        own = set(EV.loc[EV.occ_uri == o, "skill_uri"])
        CNULL[o] = np.sort(cl[~np.isin(_isk, list(own))])


def closeness_pct(o, s_):
    """percentile of closeness(o, s_) among all ESCO skills not in o's profile (NaN if s_ is linked to no occupation)"""
    if o not in CLOSE or s_ not in CLOSE[o]:
        return np.nan
    return 100.0 * np.searchsorted(CNULL[o], CLOSE[o][s_], side="left") / len(CNULL[o])


# E4b semantic plausibility with the multilingual sentence model (same model as Stage 2a)
SEM = {}
try:
    from sentence_transformers import SentenceTransformer
    _sk = pd.read_csv(esco_file("skills_en.csv"), dtype=str, keep_default_na=False,
                      usecols=["conceptUri", "preferredLabel", "description"])
    _sd = dict(zip(_sk.conceptUri, _sk.description))
    _oc = pd.read_csv(esco_file("occupations_en.csv"), dtype=str, keep_default_na=False,
                      usecols=["conceptUri", "preferredLabel", "description"])
    _od = dict(zip(_oc.conceptUri, _oc.preferredLabel + ". " + _oc.description.str[:400]))
    _m = SentenceTransformer(SEM_MODEL)
    _su = sorted(set(EV.skill_uri) | set(ADD.skill_uri))
    _se = _m.encode([f"{slabel.get(x, '')}. {str(_sd.get(x, ''))[:300]}" for x in _su], batch_size=128,
                    normalize_embeddings=True, show_progress_bar=False)
    _oe = _m.encode([_od.get(o, olab[o]) for o in OCC], normalize_embeddings=True, show_progress_bar=False)
    _sx = {x: i for i, x in enumerate(_su)}
    for k, o in enumerate(OCC):
        ref = np.sort(_se[[_sx[x] for x in EV.loc[EV.occ_uri == o, "skill_uri"]]] @ _oe[k])
        for x in ADD.loc[ADD.occ_uri == o, "skill_uri"]:
            SEM[(o, x)] = 100.0 * np.searchsorted(ref, float(_se[_sx[x]] @ _oe[k])) / len(ref)
    SEM_STATUS = "computed"
except Exception as _ex:
    SEM_STATUS = f"NOT computed ({_ex.__class__.__name__}: {str(_ex)[:120]})"
    print("\n*** WARNING: semantic plausibility (E4b) could not be computed: " + SEM_STATUS +
          "\n*** Additions are then checked only structurally. Install with:  pip install sentence-transformers\n")


say("ZenESCO - Stage 9 report: final taxonomy adjustment")
say("=" * 70)
say(f"Analysed occupations: {len(OCC)} | official links: {len(EV):,} | addition candidates: {len(ADD):,} | "
    f"non-inferiority margin m_U = {m_U:.5f}")

# ============================================================
# 2. E6 STABILITY: bootstrap the H2 2024 ads of each occupation
# ============================================================
log(f"Stability: {B} bootstrap resamples of the H2 2024 ads")
cand = pd.concat([
    EV[EV.action.isin(["DELETE candidate", "DEMOTE candidate", "UPGRADE candidate"])][
        ["occ_uri", "skill_uri", "action", "p_s", "N_null", "R_opt", "R_ess"]],
    ADD[["occ_uri", "skill_uri", "action", "p_s", "N_null", "R_opt", "R_ess"]]], ignore_index=True)
h2 = ads[ads.window == "H2_2024"]
stab = {}
for o, g in cand.groupby("occ_uri"):
    ao = h2[h2.occ_uri == o].ad_id.values
    if len(ao) == 0:
        continue
    ai = {a: i for i, a in enumerate(ao)}
    sk = list(g.skill_uri)
    si = {s: j for j, s in enumerate(sk)}
    m = ask[ask.ad_id.isin(ai) & ask.skill_uri.isin(si)]
    X = sparse.csr_matrix((np.ones(len(m)), (m.ad_id.map(ai).values, m.skill_uri.map(si).values)),
                          shape=(len(ao), len(sk)))
    X.data[:] = 1.0
    Wb = rng.multinomial(len(ao), np.ones(len(ao)) / len(ao), size=B)          # B x N_o resample weights
    nb = np.asarray(sparse.csr_matrix(Wb) @ X.todense())                       # B x skills
    N = len(ao)
    ps = g.p_s.clip(lower=1e-5).values
    a = nb + PRIOR_STRENGTH * ps
    bb = N - nb + PRIOR_STRENGTH * (1 - ps)
    pr_lift = stats.beta.sf(ps, a, bb)
    holds = np.zeros_like(pr_lift, dtype=bool)
    act = g.action.values
    weak = (stats.beta.cdf(g.N_null.values, a, bb) >= CRED) & (pr_lift < 0.5)
    up = (stats.beta.sf(g.R_ess.values, a, bb) >= CRED) & (pr_lift >= CRED)
    add = (stats.beta.sf(g.R_opt.values, a, bb) >= CRED) & (pr_lift >= CRED)
    for j, ac in enumerate(act):
        holds[:, j] = weak[:, j] if ac in ("DELETE candidate", "DEMOTE candidate") else (
            up[:, j] if ac == "UPGRADE candidate" else add[:, j])
    for j, s in enumerate(sk):
        stab[(o, s, act[j])] = float(holds[:, j].mean())
cand["stability"] = [stab.get((o, s, a), np.nan) for o, s, a in zip(cand.occ_uri, cand.skill_uri, cand.action)]

# ============================================================
# 3. DECISIONS
# ============================================================
log("Decisions")
eff = EF.set_index(["occ_uri", "skill_uri", "action"])


def efget(o, s, a, col):
    try:
        return eff.loc[(o, s, a), col]
    except KeyError:
        return np.nan


def conf(st, h1):
    return "Firm" if (st >= STAB_FIRM and h1) else "Tentative"


rows = []
for r in EV.itertuples():
    act = r.action
    st = stab.get((r.occ_uri, r.skill_uri, act), np.nan)
    dU = efget(r.occ_uri, r.skill_uri, act, "dU")
    ni = efget(r.occ_uri, r.skill_uri, act, "noninferior")
    if act == "retain: protected":
        rec, why, cf = "Retain", "protected: " + ("safety/legal" if r.prot_safety else "last link" if r.prot_last_link
                                                 else "transversal" if r.prot_transversal else "regulated"), ""
    elif act == "retain: no evidence":
        rec, why, cf = "Retain", "no evidence: absence not informative", ""
    elif act == "retain: implied":
        rec, why, cf = "Retain", "implied by: " + ", ".join(slabel.get(x, "") for x in str(r.implied_by).split("|") if x)[:80], ""
    elif act in ("DELETE candidate", "DEMOTE candidate", "UPGRADE candidate"):
        h1 = (r.lift_h1 < 1) if act != "UPGRADE candidate" else (r.lift_h1 > 1)
        h1 = bool(h1) if pd.notna(r.lift_h1) else act != "UPGRADE candidate"     # no H1 mention = still weak
        target = {"DELETE candidate": "Delete", "DEMOTE candidate": "Upgrade -> optional",
                  "UPGRADE candidate": "Upgrade -> essential"}[act]
        found = act != "UPGRADE candidate" and (
            (r.skill_type == "knowledge" and r.generality >= FOUND_KNOW) or r.generality >= FOUND_ANY)
        if found:
            rec, why, cf = "Retain", (f"no evidence: foundational/general skill (used by {int(r.generality)} ESCO "
                                      "occupations), rarely written in ads"), ""
        elif not bool(ni):
            rec, why, cf = "Expert review", "conflict: held-out identification drops beyond the margin", ""
        elif pd.isna(st) or st < STAB_LOW:
            rec, why, cf = "Retain", f"unstable evidence (stability {st:.2f})", ""
        elif st < STAB_OK:
            rec, why, cf = "Expert review", f"conflict: stability {st:.2f} (0.5-0.8)", ""
        else:
            rec = target
            why = {"Delete": "weak support: no more common than an unrelated skill, not specific",
                   "Upgrade -> optional": "weak support for an essential link (never deleted automatically)",
                   "Upgrade -> essential": "demanded as often as the occupation's essential links, and specific"}[target]
            cf = conf(st, h1)
    else:
        rec, why, cf = "Retain", "supported" if r.status == "supported" else "adequate / intermediate evidence", ""
    rows.append({"occupation": r.occupation, "cluster": r.cluster, "skill": r.skill, "skill_uri": r.skill_uri,
                 "occ_uri": r.occ_uri, "in_esco": True, "esco_relation": r.relation, "recommendation": rec,
                 "confidence": cf, "reason": why, "share_ads_H2": r.prev_raw, "share_ads_H1": r.prev_raw_h1,
                 "lift": r.lift, "pr_specific": r.pr_lift, "evidence_status": r.status, "generality": r.generality,
                 "protected": bool(r.protected_any), "semantic_pct": r.sem_pct, "stability": st,
                 "dMRR_heldout": dU, "dDistinct_o": efget(r.occ_uri, r.skill_uri, act, "dDist_o"),
                 "dCoverage_o": efget(r.occ_uri, r.skill_uri, act, "dDC_o"), "screening": act})
for r in ADD.itertuples():
    st = stab.get((r.occ_uri, r.skill_uri, "ADD candidate"), np.nan)
    dU = efget(r.occ_uri, r.skill_uri, "ADD candidate", "dU")
    imp = bool(efget(r.occ_uri, r.skill_uri, "ADD candidate", "improves"))
    h1 = bool(r.same_direction_h1) if pd.notna(r.same_direction_h1) else False
    sem_p = SEM.get((r.occ_uri, r.skill_uri), r.sem_pct)
    lowp = pd.notna(sem_p) and sem_p < SEM_LOW
    cpct = closeness_pct(r.occ_uri, r.skill_uri)
    struct_ok = pd.notna(cpct) and cpct >= STRUCT_PCT
    if not struct_ok:
        rec, why, cf = "Expert review", ("low plausibility (structural): the ESCO occupations using this skill are no "
                                         "closer to this occupation than a random skill's (possible extraction error "
                                         "or boilerplate)"), ""
    elif lowp:
        rec, why, cf = "Expert review", f"low plausibility (semantic): percentile {sem_p:.0f} of the occupation's own links", ""
    elif imp and pd.notna(st) and st >= STAB_OK and h1:
        rec = "Add (essential)" if r.proposed_type == "essential" else "Add (optional)"
        why, cf = "emerging demand: common, specific, improves held-out identification", conf(st, h1)
    elif pd.notna(dU) and dU < m_U:
        rec, why, cf = "Reject addition", "harms held-out identification", ""
    else:
        parts = [p for p, c in [("no clear held-out gain", not imp), (f"stability {st:.2f}", not (pd.notna(st) and st >= STAB_OK)),
                                ("H1 2025 disagrees", not h1)] if c]
        rec, why, cf = "Expert review", "conflict: " + ", ".join(parts), ""
    rows.append({"occupation": r.occupation, "cluster": ocl.get(r.occ_uri), "skill": r.skill,
                 "skill_uri": r.skill_uri, "occ_uri": r.occ_uri, "in_esco": False, "esco_relation": "",
                 "recommendation": rec, "confidence": cf, "reason": why, "share_ads_H2": r.prev_raw,
                 "share_ads_H1": r.prev_raw_h1, "lift": r.lift, "pr_specific": r.pr_lift,
                 "evidence_status": "not in ESCO", "generality": gen.get(r.skill_uri, 0), "protected": False,
                 "semantic_pct": sem_p, "stability": st, "dMRR_heldout": dU,
                 "dDistinct_o": efget(r.occ_uri, r.skill_uri, "ADD candidate", "dDist_o"),
                 "dCoverage_o": efget(r.occ_uri, r.skill_uri, "ADD candidate", "dDC_o"), "screening": "ADD candidate",
                 "structural_pct": cpct})
R1 = pd.DataFrame(rows)
CHANGE = ["Delete", "Upgrade -> optional", "Upgrade -> essential", "Add (essential)", "Add (optional)"]

# ============================================================
# 4. PROFILES: original, ZenESCO (Firm + Tentative), Firm only, frequency pruning
# ============================================================
def build(which):
    P = {o: {} for o in OCC}
    for r in EV.itertuples():
        P[r.occ_uri][r.skill_uri] = 1.0 if r.relation == "essential" else ALPHA
    if which == "original":
        return P
    if which == "frequency pruning":
        for r in EV.itertuples():
            if r.n == 0:
                P[r.occ_uri].pop(r.skill_uri, None)
        return P
    sel = R1[R1.recommendation.isin(CHANGE)]
    if which == "firm":
        sel = sel[sel.confidence == "Firm"]
    for r in sel.itertuples():
        if r.recommendation == "Delete":
            P[r.occ_uri].pop(r.skill_uri, None)
        elif r.recommendation in ("Upgrade -> optional", "Add (optional)"):
            P[r.occ_uri][r.skill_uri] = ALPHA
        else:
            P[r.occ_uri][r.skill_uri] = 1.0
    return P


PROF = {k: build(k) for k in ["original", "zenesco", "firm", "frequency pruning"]}

# ============================================================
# 5. HELD-OUT EVALUATION (H1 2025), same scoring as Stage 7
# ============================================================
log("Held-out evaluation")
test = ads[ads.window == "H1_2025"].reset_index(drop=True)
at = ask[ask.ad_id.isin(set(test.ad_id))]
SK = sorted(set(at.skill_uri) | {s for P in PROF.values() for p in P.values() for s in p})
si = {s: i for i, s in enumerate(SK)}
aid = {a: i for i, a in enumerate(test.ad_id)}
X = sparse.csr_matrix((np.ones(len(at)), (at.ad_id.map(aid).values, at.skill_uri.map(si).values)),
                      shape=(len(test), len(SK)))
X.data[:] = 1.0
idf = np.maximum(np.array([np.log(N_OCC / gen.get(s, 1)) for s in SK]), 0.05)
oi = {o: i for i, o in enumerate(OCC)}
true = test.occ_uri.map(oi).values
nnz = np.asarray(X.sum(axis=1)).ravel()


def matrix(P):
    W = np.zeros((len(OCC), len(SK)))
    for o, p in P.items():
        for s, w in p.items():
            W[oi[o], si[s]] = w
    return W


Xn = (sparse.diags(1.0 / np.maximum(np.sqrt(np.asarray(X.multiply(X).multiply(idf ** 2).sum(axis=1)).ravel()), 1e-12))
      @ X.multiply(idf).tocsr()).tocsr()


def prof_vecs(P):
    V = matrix(P) * idf
    return V / np.maximum(np.sqrt((V ** 2).sum(axis=1)), 1e-12)[:, None]


def acc_rows(P, mask):
    S = np.asarray(Xn[mask] @ prof_vecs(P).T)
    t = true[mask]
    ts = S[np.arange(len(S)), t]
    return (((S > ts[:, None]).sum(axis=1) == 0) & ((S == ts[:, None]).sum(axis=1) == 1) & (ts > 0)).astype(float)


def score(P):
    Vn = prof_vecs(P)
    S = np.asarray(Xn @ Vn.T)
    ts = S[np.arange(len(S)), true]
    acc = ((S > ts[:, None]).sum(axis=1) == 0) & ((S == ts[:, None]).sum(axis=1) == 1) & (ts > 0)
    rr = 1.0 / (1 + (S > ts[:, None]).sum(axis=1))
    C = Vn @ Vn.T
    np.fill_diagonal(C, -1)
    return {"acc": acc.astype(float), "rr": rr, "sim": ts, "dist": 1 - C.max(axis=1)}


SC = {k: score(P) for k, P in PROF.items()}

h1prev = (at.merge(test[["ad_id", "occ_uri"]], on="ad_id").groupby(["occ_uri", "skill_uri"]).ad_id.nunique()
          / test.groupby("occ_uri").size()).rename("p").reset_index()


def topk_cov(o, P):
    h = h1prev[(h1prev.occ_uri == o) & ~h1prev.skill_uri.isin(lang)].nlargest(TOP_K, "p")
    return h.skill_uri.isin(set(P[o])).mean() if len(h) else np.nan


def boot_diff(v1, v0, n=1000):
    if len(v1) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(v1), size=(n, len(v1)))
    d = v1[idx].mean(axis=1) - v0[idx].mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def occ_drop(P_new):
    """occupations whose held-out identification drops significantly (95% bootstrap CI below 0)"""
    a1, a0 = score(P_new)["acc"], SC["original"]["acc"]
    bad = []
    for o in OCC:
        m = true == oi[o]
        lo, hi = boot_diff(a1[m], a0[m], n=400)
        if hi < 0:
            bad.append((a1[m].mean() - a0[m].mean(), o))
    return sorted(bad)


log("Joint guard: no occupation may become significantly harder to recognise")
guard_log = []
for it in range(MAX_GUARD):
    bad = occ_drop(PROF["zenesco"])
    if not bad:
        break
    _, ob = bad[0]
    mb = true == oi[ob]
    base_acc = acc_rows(PROF["zenesco"], mb).mean()
    sk_ob = set(at[at.ad_id.isin(set(test.ad_id[mb]))].skill_uri)
    ch_ = R1[R1.recommendation.isin(CHANGE) & ((R1.occ_uri == ob) | R1.skill_uri.isin(sk_ob))]
    best = None
    for idx, r in ch_.iterrows():
        P2 = {o: dict(p) for o, p in PROF["zenesco"].items()}
        orig_w = PROF["original"][r.occ_uri].get(r.skill_uri)
        if orig_w is None:
            P2[r.occ_uri].pop(r.skill_uri, None)
        else:
            P2[r.occ_uri][r.skill_uri] = orig_w
        gain = acc_rows(P2, mb).mean() - base_acc
        if best is None or gain > best[0]:
            best = (gain, idx, P2)
    if best is None or best[0] <= 0:
        break
    gain, idx, P2 = best
    R1.loc[idx, ["recommendation", "confidence", "reason"]] = [
        "Expert review", "", f"conflict: joint edits lower held-out identification of {olab[ob]}"]
    guard_log.append((olab[ob], R1.loc[idx, "occupation"], R1.loc[idx, "skill"], gain))
    PROF["zenesco"] = P2
    PROF["firm"] = build("firm")
    SC["zenesco"], SC["firm"] = score(PROF["zenesco"]), score(PROF["firm"])

# ============================================================
# 5b. EXPERT REVIEW (Stage 10) - applied only if BASE\stage10\review_results.xlsx exists
#     final decision = user_recommendation where user_agree = N, otherwise "Final recommendation"
# ============================================================
REVIEW = BASE / "stage10" / "review_results.xlsx"
if not REVIEW.exists() and (BASE / "stage10" / "review_results.csv").exists():
    REVIEW = BASE / "stage10" / "review_results.csv"


def read_review(path):
    """Read the 'Review' sheet. Tries pandas+openpyxl, then openpyxl directly, then a CSV export of the sheet."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=None, engine="python")
    errors = []
    try:
        return pd.read_excel(path, sheet_name="Review", dtype=str, engine="openpyxl").fillna("")
    except Exception as e:
        errors.append(f"pandas.read_excel: {e!r}")
    try:
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Review"]
        rows = [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception as e:
        errors.append(f"openpyxl: {e!r}")
    sys.exit("Could not read review_results.xlsx:\n  " + "\n  ".join(errors) + "\n"
             f"Python used: {sys.executable}\n"
             "Fix 1: install openpyxl for THIS Python:  \"" + sys.executable + "\" -m pip install openpyxl\n"
             "Fix 2: open review_results.xlsx in Excel, sheet 'Review' -> Save As -> CSV UTF-8, "
             "name it review_results.csv in the same stage10 folder, and run again.")
R1["expert_review"] = ""
review_lines, review_drop = [], []
VALID_ESCO = {"essential": {"Retain", "Delete", "Upgrade -> optional"},
              "optional": {"Retain", "Delete", "Upgrade -> essential"}}
VALID_ADD = {"Add (essential)", "Add (optional)", "Reject addition"}
if REVIEW.exists():
    log("Applying expert review (Stage 10)")
    rv = read_review(REVIEW)
    rv["agree"] = rv["user_agree"].str.strip().str.upper()
    rv["mine"] = rv["Final recommendation"].str.strip()
    rv["user"] = np.where(rv.agree == "N", rv["user_recommendation"].str.strip(), rv.mine)
    # "Delete" on a candidate means "Reject addition", and "Reject addition" on an ESCO link means "Delete"
    cand_ = rv["ESCO now"].str.strip() == "not in ESCO"
    rv.loc[cand_ & (rv.user == "Delete"), "user"] = "Reject addition"
    rv.loc[~cand_ & (rv.user == "Reject addition"), "user"] = "Delete"
    allowed = [VALID_ADD if c else VALID_ESCO.get(e.strip(), set()) for c, e in zip(cand_, rv["ESCO now"])]
    rv["user_ok"] = [u in a for u, a in zip(rv.user, allowed)]
    for r in rv[(rv.agree == "N") & ~rv.user_ok].to_dict("records"):
        say(f"  WARNING {r['item_id']} ({r['Skill']}): your decision '{r['user']}' is not possible for "
            f"{'a candidate' if r['ESCO now'] == 'not in ESCO' else 'an ESCO ' + r['ESCO now'] + ' link'} - "
            f"kept '{r['mine']}' and left out of the agreement")
    rv["applied"] = np.where((rv.agree == "N") & (rv.user != "") & rv.user_ok, rv.user, rv.mine)
    pos = {(o, s): i for i, (o, s) in zip(R1.index, zip(R1.occ_uri, R1.skill_uri))}
    n_app, n_bad = 0, 0
    for r in rv.to_dict("records"):
        i = pos.get((r["occ_uri"], r["skill_uri"]))
        if i is None:
            continue
        new, old = r["applied"], R1.at[i, "recommendation"]
        ok = new in (VALID_ESCO.get(R1.at[i, "esco_relation"], set()) if R1.at[i, "in_esco"] else VALID_ADD)
        if not ok:
            n_bad += 1
            say(f"  WARNING {r['item_id']}: '{new}' is not possible for this item - ignored")
            continue
        if old == "Expert review":
            tag = "resolved"
        else:
            tag = "confirmed" if new == old else "overruled"
        R1.loc[i, ["recommendation", "expert_review"]] = [new, tag]
        if tag != "confirmed":
            R1.at[i, "confidence"] = "Expert" if new in CHANGE else ""
        R1.at[i, "reason"] = f"{R1.at[i, 'reason']} | expert review: {r.get('Reason code', '')}"
        n_app += 1
    PROF["zenesco"], PROF["firm"] = build("zenesco"), build("firm")
    SC["zenesco"], SC["firm"] = score(PROF["zenesco"]), score(PROF["firm"])
    # agreement between the analysis assistant and the human validator
    v = rv[rv.agree.isin(["Y", "N"]) & (rv.user != "") & rv.user_ok]
    review_lines.append(f"Expert review applied to {n_app} items ({n_bad} invalid entries ignored): "
                        + ", ".join(f"{k} {c}" for k, c in R1.expert_review[R1.expert_review != ""].value_counts().items()))
    if len(v):
        dis = v[v.mine != v.user]
        review_lines.append(f"Expert verification: {len(v)} decisions of the assistant checked by the domain expert, "
                            f"{len(dis)} changed ({len(dis) / len(v):.1%})")
        for r in dis.to_dict("records"):
            review_lines.append(f"  changed {r['item_id']}: {r['Occupation']} / {r['Skill']}: assistant "
                                f"'{r['mine']}' -> expert '{r['user']}'")
    else:
        review_lines.append("Expert verification: not filled in yet (columns user_agree / user_recommendation are empty)")
else:
    review_lines.append("Expert review: not applied (no stage10\\review_results.xlsx)")


# ============================================================
# 6. R2
# ============================================================
r2 = []
for o in OCC:
    g = R1[R1.occ_uri == o]
    off = g[g.in_esco]
    m = true == oi[o]
    lo, hi = boot_diff(SC["zenesco"]["acc"][m], SC["original"]["acc"][m])
    vc = g.recommendation.value_counts()
    r2.append({
        "cluster": ocl[o], "occupation": olab[o],
        "ESCO links (ess/opt)": f"{(off.esco_relation == 'essential').sum()} / {(off.esco_relation == 'optional').sum()}",
        "Retained": int(vc.get("Retain", 0)), "Deleted": int(vc.get("Delete", 0)),
        "Ess->opt": int(vc.get("Upgrade -> optional", 0)), "Opt->ess": int(vc.get("Upgrade -> essential", 0)),
        "Added": int(vc.get("Add (essential)", 0) + vc.get("Add (optional)", 0)),
        "Expert review": int(vc.get("Expert review", 0)), "Rejected additions": int(vc.get("Reject addition", 0)),
        "Resolved by expert review": int((g.expert_review == "resolved").sum()),
        "Firm changes": int(((g.confidence == "Firm") & g.recommendation.isin(CHANGE)).sum()),
        "Profile size before": len(PROF["original"][o]), "Profile size after": len(PROF["zenesco"][o]),
        f"Top-{TOP_K} held-out demand covered before": round(topk_cov(o, PROF["original"]) * 100),
        f"Top-{TOP_K} held-out demand covered after": round(topk_cov(o, PROF["zenesco"]) * 100),
        "Identification before": round(SC["original"]["acc"][m].mean(), 3),
        "Identification after": round(SC["zenesco"]["acc"][m].mean(), 3),
        "Identification change 95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
        "Distinctiveness before": round(SC["original"]["dist"][oi[o]], 3),
        "Distinctiveness after": round(SC["zenesco"]["dist"][oi[o]], 3),
        "Held-out ads": int(m.sum())})
R2 = pd.DataFrame(r2).sort_values(["cluster", "occupation"])
tot = {"cluster": "", "occupation": "All analysed occupations",
       "ESCO links (ess/opt)": f"{(R1.in_esco & (R1.esco_relation == 'essential')).sum()} / {(R1.in_esco & (R1.esco_relation == 'optional')).sum()}"}
for c in ["Retained", "Deleted", "Ess->opt", "Opt->ess", "Added", "Expert review", "Rejected additions",
          "Resolved by expert review", "Firm changes",
          "Profile size before", "Profile size after", "Held-out ads"]:
    tot[c] = int(R2[c].sum())
for c in [f"Top-{TOP_K} held-out demand covered before", f"Top-{TOP_K} held-out demand covered after",
          "Distinctiveness before", "Distinctiveness after"]:
    tot[c] = round(R2[c].mean(), 3 if "Distinct" in c else 0)
lo, hi = boot_diff(SC["zenesco"]["acc"], SC["original"]["acc"])
tot.update({"Identification before": round(SC["original"]["acc"].mean(), 3),
            "Identification after": round(SC["zenesco"]["acc"].mean(), 3),
            "Identification change 95% CI": f"[{lo:+.3f}, {hi:+.3f}]"})
R2 = pd.concat([R2, pd.DataFrame([tot])], ignore_index=True)
if REVIEW.exists():   # same intervals as Table R2
    hi_ = R2["Identification change 95% CI"].str.extract(r",\s*([+-][0-9.]+)\]")[0].astype(float)
    worse = R2.occupation[(hi_ < 0) & (R2.occupation != "All analysed occupations")].tolist()
    review_lines.append("Occupations significantly harder to recognise after the expert decisions (Table R2 interval "
                        "below 0; expert decisions are not reverted): " + ", ".join(worse) if worse else
                        "No occupation becomes significantly harder to recognise after the expert decisions (Table R2)")

# ============================================================
# 7. R1 main-text excerpt: the strongest changes per occupation
# ============================================================
ch = R1[R1.recommendation.isin(CHANGE)].copy()
ch["type"] = ch.recommendation.replace({"Add (essential)": "Add", "Add (optional)": "Add"})
ch["order"] = np.where(ch.type == "Delete", -ch.stability.fillna(0) + ch.share_ads_H2,
                       -ch.share_ads_H2.fillna(0))
ch["crank"] = ch.confidence.map({"Firm": 0, "Expert": 1, "Tentative": 2}).fillna(3)
ex = (ch.sort_values(["occupation", "type", "crank", "order"], ascending=[True, True, True, True])
        .groupby(["occupation", "type"]).head(2))
fmt = lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%"
MAIN = pd.DataFrame({
    "Occupation": ex.occupation, "Skill": ex.skill,
    "ESCO now": np.where(ex.in_esco, ex.esco_relation, "not linked"),
    "Recommendation": ex.recommendation,
    "Share of ads H2 2024 / H1 2025": [f"{fmt(a)} / {fmt(b)}" for a, b in zip(ex.share_ads_H2, ex.share_ads_H1)],
    "Specific (Pr)": ex.pr_specific.round(2), "Held-out dMRR x1000": (ex.dMRR_heldout * 1000).round(2),
    "Stability": ex.stability.round(2), "Confidence": ex.confidence})
MAIN = MAIN.sort_values(["Occupation", "Recommendation"])

UPD = pd.DataFrame([{"occupation": olab[o], "skill": slabel.get(s, s), "skill_uri": s,
                     "relation": "essential" if w == 1.0 else "optional",
                     "change": ("added" if s not in PROF["original"][o] else
                                "upgraded" if w > PROF["original"][o][s] else
                                "demoted" if w < PROF["original"][o][s] else "")}
                    for o in OCC for s, w in sorted(PROF["zenesco"][o].items(), key=lambda t: (-t[1], slabel.get(t[0], "")))])

# ============================================================
# 8. REPORT
# ============================================================
say()
say("Recommendations (existing ESCO links | addition candidates)")
vc = pd.crosstab(R1.recommendation, R1.in_esco.map({True: "ESCO link", False: "candidate"}))
say(vc.to_string())
say("\nConfidence of the changes: " + ", ".join(f"{k} {v}" for k, v in ch.confidence.value_counts().items()))
say(f"Stability of candidate edits: median {cand.stability.median():.2f}; share >= 0.8: {(cand.stability >= STAB_OK).mean():.1%}")
say(f"\nJoint guard: {len(guard_log)} change(s) sent to expert review because together they made another "
    "occupation significantly harder to recognise")
for ob, oc, sk_, gn in guard_log:
    say(f"  protects {ob}: {oc} / {sk_}  (+{gn:.3f} accuracy for {ob})")
for line in review_lines:
    say(line)
say(f"Retained because foundational/general (absence not informative): "
    f"{R1.reason.str.startswith('no evidence: foundational').sum()}")
say(f"Semantic plausibility (E4b): {SEM_STATUS}")
say(f"Additions sent to expert review for low plausibility (rule A3): structural "
    f"{R1.reason.str.startswith('low plausibility (structural)').sum()}, semantic "
    f"{R1.reason.str.startswith('low plausibility (semantic)').sum()}, of {(~R1.in_esco).sum()} candidates")
pl = R1[~R1.in_esco & R1.reason.str.startswith("low plausibility")].sort_values("share_ads_H2", ascending=False)
say("  most demanded of these: " + "; ".join(f"{o} / {k} ({v:.0%})" for o, k, v in
                                             zip(pl.occupation.head(10), pl.skill.head(10), pl.share_ads_H2.head(10))))
say("\nTable R2 - per occupation")
with pd.option_context("display.width", 320, "display.max_columns", 30):
    say(R2.drop(columns=["cluster"]).to_string(index=False))
say("\nHeld-out identification accuracy (H1 2025 ads, all analysed occupations)")
for k in ["original", "frequency pruning", "firm", "zenesco"]:
    say(f"  {k:18s} {SC[k]['acc'].mean():.3f}   mean ad-profile similarity {SC[k]['sim'].mean():.3f}   "
        f"median distinctiveness {np.median(SC[k]['dist']):.3f}")
say("\nTable R1 excerpt (main text)")
with pd.option_context("display.width", 320, "display.max_colwidth", 40):
    say(MAIN.to_string(index=False))

# ============================================================
# 9. SAVE
# ============================================================
OUT.mkdir(parents=True, exist_ok=True)
R1.sort_values(["occupation", "in_esco", "recommendation"]).to_csv(OUT / "R1_all_links.csv", index=False, encoding="utf-8-sig")
MAIN.to_csv(OUT / "R1_main_text.csv", index=False, encoding="utf-8-sig")
R2.to_csv(OUT / "R2_occupation_summary.csv", index=False, encoding="utf-8-sig")
UPD.to_csv(OUT / "updated_profiles.csv", index=False, encoding="utf-8-sig")
try:
    with pd.ExcelWriter(OUT / "ZenESCO_tables.xlsx") as xw:
        R2.to_excel(xw, sheet_name="R2 occupation summary", index=False)
        MAIN.to_excel(xw, sheet_name="R1 main text", index=False)
        R1.drop(columns=["occ_uri"]).to_excel(xw, sheet_name="R1 all links", index=False)
        UPD.to_excel(xw, sheet_name="Updated profiles", index=False)
    log("Excel workbook saved")
except Exception as ex_:
    log(f"Excel skipped ({ex_.__class__.__name__}: pip install openpyxl); CSV files are complete")

# ============================================================
# 10. FIGURE F8 - alignment of the original vs updated taxonomy
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    COL = {"original": "#2a78d6", "zenesco": "#008300", "frequency pruning": "#eb6834"}
    NAME = {"original": "original ESCO", "zenesco": "ZenESCO", "frequency pruning": "frequency pruning"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.62, width_ratios=[1.1, 1])
    # (a) ECDF of ad-profile similarity with simultaneous 95% bands (DKW)
    ax = fig.add_subplot(gs[0, 0])
    n = len(test)
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n))
    grid_ = np.linspace(0, max(SC[k]["sim"].max() for k in COL), 400)
    E = {}
    for k in ["frequency pruning", "original", "zenesco"]:
        v = np.sort(SC[k]["sim"])
        E[k] = np.searchsorted(v, grid_, side="right") / n
        ax.plot(grid_, E[k], color=COL[k], lw=1.4 if k == "zenesco" else 1.0, label=NAME[k])
        ax.fill_between(grid_, np.clip(E[k] - eps, 0, 1), np.clip(E[k] + eps, 0, 1), color=COL[k], alpha=0.12, lw=0)
    better = (E["zenesco"] + eps) < (E["original"] - eps)
    if better.any():
        ax.fill_between(grid_, 0, 1, where=better, color="#008300", alpha=0.06, lw=0, transform=ax.get_xaxis_transform())
        ax.text(grid_[better].mean(), 0.06, "ZenESCO higher\n(5% FWER)", ha="center", fontsize=6, color="#008300")
    ks = stats.ks_2samp(SC["zenesco"]["sim"], SC["original"]["sim"])
    ax.set_xlabel("similarity of a held-out ad to its occupation's profile")
    ax.set_ylabel("cumulative share of held-out ads")
    ax.set_title("a  Alignment of every held-out ad (H1 2025)", loc="left", fontsize=8.3, fontweight="bold")
    ax.text(0.98, 0.3, f"KS = {ks.statistic:.2f}, p = {ks.pvalue:.1e}\nn = {n:,} ads", transform=ax.transAxes,
            ha="right", fontsize=6.2, color=INK2)
    ax.legend(loc="lower right", fontsize=6.4)
    ax.grid(color=GRID, lw=0.5)
    # (b) identification per occupation, before -> after
    bx = fig.add_subplot(gs[:, 1])
    d = R2[R2.occupation != "All analysed occupations"].copy()
    d["fp"] = [SC["frequency pruning"]["acc"][true == oi[o]].mean()
               for o in [next(u for u in OCC if olab[u] == x) for x in d.occupation]]
    d = d.sort_values("Identification after")
    y = np.arange(len(d))
    for yi, a_, b_ in zip(y, d["Identification before"], d["Identification after"]):
        bx.plot([a_, b_], [yi, yi], color="#008300" if b_ > a_ else GRID, lw=1.6)
    bx.scatter(d["Identification before"], y, s=24, c="white", edgecolors=COL["original"], lw=1.1, zorder=3, label="original ESCO")
    bx.scatter(d.fp, y, s=18, marker="|", c=COL["frequency pruning"], lw=1.4, zorder=3, label="frequency pruning")
    bx.scatter(d["Identification after"], y, s=24, c=COL["zenesco"], zorder=4, label="ZenESCO")
    bx.set_yticks(y)
    import textwrap
    bx.set_yticklabels([textwrap.fill(t_, 20) for t_ in d.occupation], fontsize=6.2, linespacing=0.95)
    bx.set_xlabel(f"identification accuracy on held-out ads (chance = {1 / len(OCC):.2f})")
    bx.axvline(1 / len(OCC), color=MUTED, lw=0.6, ls=(0, (2, 2)))
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.legend(loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=3, fontsize=6.0, handletextpad=0.2, columnspacing=0.8)
    bx.set_title("b  Recognising each occupation", loc="left", fontsize=8.3, fontweight="bold")
    # (c) boxplots by occupation level, before vs after
    cx = fig.add_subplot(gs[1, 0])
    lvl = test.occ_uri.map(lambda u: {"1": "Managers\n(ISCO 1)", "2": "Professionals\n(ISCO 2)",
                                      "3": "Technicians\n(ISCO 3)"}.get(str(isco.get(u, ""))[:1], "Other"))
    groups = [g_ for g_ in ["Managers\n(ISCO 1)", "Professionals\n(ISCO 2)", "Technicians\n(ISCO 3)"] if (lvl == g_).any()]
    pos, data, cols = [], [], []
    for i, g_ in enumerate(groups):
        for j, k in enumerate(["original", "zenesco"]):
            pos.append(i * 3 + j)
            data.append(SC[k]["sim"][(lvl == g_).values])
            cols.append(COL[k])
    bp = cx.boxplot(data, positions=pos, widths=0.75, notch=False, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=0.9))
    for p__, c_ in zip(bp["boxes"], cols):
        p__.set_facecolor(c_)
        p__.set_alpha(0.55)
    cx.set_xticks([i * 3 + 0.5 for i in range(len(groups))])
    cx.set_xticklabels(groups, fontsize=6.6)
    cx.set_ylabel("ad-profile similarity")
    cx.legend(handles=[Line2D([], [], marker="s", ls="", color=COL[k], alpha=0.6, ms=6, label=NAME[k])
                       for k in ["original", "zenesco"]], loc="upper left", ncol=2, fontsize=6.2)
    cx.set_ylim(top=cx.get_ylim()[1] * 1.15)
    cx.grid(axis="y", color=GRID, lw=0.5)
    cx.set_title("c  By occupation level", loc="left", fontsize=8.3, fontweight="bold")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"F8_alignment_before_after.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    log("F8 saved")
except Exception as ex_:
    import traceback
    traceback.print_exc()
    log(f"F8 skipped: {ex_!r}")

say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "stage9_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
