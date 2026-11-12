"""
ZenESCO - Stage 4: labour-market evidence for every ESCO link, and missing links
================================================================================

Objective
    For each analysed occupation o and each skill s, measure what the job ads say:
      E1 demand        posterior prevalence p_os (Beta-Binomial)
      E2 specificity   lift = p_os / p_s and Pr(p_os > p_s), p_s = share of ALL ads mentioning s
      E3 observability is the skill visible to the extractor, and would its absence be significant?
      E4 plausibility  semantic similarity of s to o, as a percentile of o's own ESCO links (optional)
    Then give every official link an EVIDENCE STATUS and list empirical-only links (addition candidates).
    Nothing is deleted or added here: Stage 4 is evidence only. Decisions come in Stages 6-9.

Windows
    Selection = H2 2024 (all evidence used for screening). H1 2025 is computed the same way and kept
    only as a direction check (persistence); the held-out tests of Stage 7 reweight H1 to H2's portal mix.

Reference levels per occupation (no fixed cut-offs on frequency; see the decision rules in README.md)
    N_o       95th percentile of the prevalence, in o's ads, of skills UNRELATED to o
              (visible skills not linked to any ESCO occupation in o's ISCO minor group)
    R_opt(o)  median prevalence of o's SUPPORTED optional ESCO links (what an optional skill that the market
              confirms looks like in o's ads); R_ess(o) the same for essential links; R_dem(o) for all
              supported links (used for the power test). If fewer than MIN_REF supported links of a type
              exist, all of o's supported links are used.
              (First run, 27 Oct 2026: the median over ALL visible links was 0 for most occupations, because
              most ESCO links are never mentioned in ads; that made the power and addition tests meaningless.
              Both medians are reported: the "all links" medians are a finding, not a reference.)

Evidence status of an official link (first match wins)
    supported        Pr(p_os > p_s) >= 0.95                         (over-represented in o's ads)
    unobservable     skill not visible to the extractor (< MIN_VISIBLE ads in the window), or absence
                     not significant: Pr(X <= n_os | Bin(N_o, R_dem(o))) >= 0.05
    weak             Pr(p_os <= N_o) >= 0.95 and Pr(p_os > p_s) < 0.5  (no more common than an unrelated skill)
      contradicted   weak here, but supported in another analysed occupation (possible misattribution)
    intermediate     observed and observable, neither supported nor weak
Addition candidate (pair not in ESCO): Pr(p_os > p_s) >= 0.95 and Pr(p_os >= R_opt(o)) >= 0.95;
    flagged separately if the skill is a language skill, a transversal skill (ESCO keeps these outside
    occupation profiles by design) or a boilerplate skill.

Inputs   BASE\\stage1\\ads_clean.parquet, ad_skills_clean.parquet, boilerplate_candidates.csv
         BASE\\stage2\\ad_occupation.parquet, stage2b_targets.csv
         BASE\\stage3\\official_edges.parquet
         BASE\\esco\\ occupations_en, skills_en, occupationSkillRelations_en, collections
Outputs  BASE\\stage4\\edge_evidence.parquet         one row per official link (16 occupations)
         BASE\\stage4\\addition_candidates.parquet   empirical-only links that pass the addition screen
         BASE\\stage4\\pair_evidence.parquet         every (occupation, skill) pair seen in H2 ads
         BASE\\stage4\\occupation_evidence.csv       one row per occupation: references, status counts,
                                                     digital gap, essential/optional consistency
         BASE\\stage4\\F3_digital_gap.png, F4_evidence_status.png, stage4_report.txt
Run      python stage4_evidence.py                  # BASE = C:\\skillab_data
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
S1, S2, S3, ESCO = BASE / "stage1", BASE / "stage2", BASE / "stage3", BASE / "esco"
OUT = BASE / "stage4"

WIN_SEL, WIN_EVAL = "H2_2024", "H1_2025"
PRIOR_STRENGTH = 2.0   # Beta prior centred on the skill's market prevalence, worth 2 pseudo-ads
CRED = 0.95            # credibility level for every decision (varied in Stage 8)
NULL_Q = 0.95          # N_o = this quantile of unrelated-skill prevalence
ALPHA_POW = 0.05       # absence is informative only if it would be this unlikely at level R_dem(o)
MIN_REF = 3            # minimum supported links of a type to compute R_opt / R_ess from that type alone
MIN_VISIBLE = 5        # a skill the extractor finds in fewer ads than this in a window is "not visible"
USE_SEMANTIC = True    # E4 with the multilingual sentence model of Stage 2a (skipped if unavailable)
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

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
# 1. LOAD
# ============================================================
log("Loading stage outputs")
ads = pd.read_parquet(S1 / "ads_clean.parquet", columns=["ad_id", "portal", "window", "analysable"])
ads = ads[ads.analysable & ads.window.isin([WIN_SEL, WIN_EVAL])].drop(columns="analysable")
ads["ad_id"] = ads.ad_id.astype(str)
ask = pd.read_parquet(S1 / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(ads.ad_id))].drop_duplicates()
occ_of = pd.read_parquet(S2 / "ad_occupation.parquet", columns=["ad_id", "occ_uri", "occupation"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
tg = pd.read_csv(S2 / "stage2b_targets.csv")
A = tg[tg.analysed.astype(bool)][["occ_uri", "occupation", "cluster"]].reset_index(drop=True)
E = pd.read_parquet(S3 / "official_edges.parquet")
E = E[E.occ_uri.isin(set(A.occ_uri))].copy()

skills = read_esco("skills_en.csv")
skill_label = dict(zip(skills.conceptUri, skills.preferredLabel))
skill_desc = dict(zip(skills.conceptUri, skills.description))
coll = {}
for tag, f in [("digital", "digitalSkillsCollection_en.csv"), ("transversal", "transversalSkillsCollection_en.csv"),
               ("language", "languageSkillsCollection_en.csv")]:
    d = read_esco(f)
    coll[tag] = set(d.conceptUri) if d is not None else set()
coll["transversal"] |= set(skills.conceptUri[skills.reuseLevel == "transversal"])
occs = read_esco("occupations_en.csv")
occ_isco = dict(zip(occs.conceptUri, occs.iscoGroup))
occ_desc = dict(zip(occs.conceptUri, occs.description))
rel = read_esco("occupationSkillRelations_en.csv", usecols=["occupationUri", "skillUri"])
bp_f = S1 / "boilerplate_candidates.csv"
boiler = set(pd.read_csv(bp_f).query("candidate == True").skill_uri) if bp_f.exists() else set()

ads = ads.merge(occ_of[["ad_id", "occ_uri"]], on="ad_id", how="left")
say("ZenESCO - Stage 4 report: labour-market evidence for ESCO links")
say("=" * 72)
say(f"Analysed occupations: {len(A)} | official links: {len(E):,} "
    f"({(E.relation == 'essential').sum()} essential, {(E.relation == 'optional').sum()} optional)")
for w in [WIN_SEL, WIN_EVAL]:
    aw = ads[ads.window == w]
    say(f"  {w}: all analysable ads {len(aw):,}; ads in analysed occupations {aw.occ_uri.isin(set(A.occ_uri)).sum():,}; "
        f"skill mentions {ask.ad_id.isin(set(aw.ad_id)).sum():,}")

# ESCO-side "unrelated" sets: skills linked to any occupation in the same ISCO minor group (3 digits)
rel["minor"] = rel.occupationUri.map(lambda u: str(occ_isco.get(u, ""))[:3])
minor_skills = rel.groupby("minor").skillUri.agg(set).to_dict()


# ============================================================
# 2. EVIDENCE PER WINDOW
# ============================================================
def evidence(window):
    aw = ads[ads.window == window]
    n_all = len(aw)
    sk = ask[ask.ad_id.isin(set(aw.ad_id))].merge(aw[["ad_id", "occ_uri"]], on="ad_id")
    p_s = sk.groupby("skill_uri").ad_id.nunique() / n_all                 # market prevalence of each skill
    visible = set(p_s.index[p_s * n_all >= MIN_VISIBLE])
    N_o = aw[aw.occ_uri.isin(set(A.occ_uri))].groupby("occ_uri").size()
    cnt = (sk[sk.occ_uri.isin(set(A.occ_uri))].groupby(["occ_uri", "skill_uri"]).ad_id.nunique()
           .rename("n").reset_index())
    # all official pairs must be present, also with zero mentions
    pairs = pd.concat([cnt, E[["occ_uri", "skill_uri"]].assign(n=0)]).groupby(["occ_uri", "skill_uri"]).n.max().reset_index()
    pairs["N"] = pairs.occ_uri.map(N_o).fillna(0).astype(int)
    pairs["p_s"] = pairs.skill_uri.map(p_s).fillna(0.0)
    ps = pairs.p_s.clip(lower=0.5 / n_all)
    a = pairs.n + PRIOR_STRENGTH * ps
    b = pairs.N - pairs.n + PRIOR_STRENGTH * (1 - ps)
    pairs["prev_raw"] = pairs.n / pairs.N.clip(lower=1)
    pairs["prev"] = a / (a + b)
    pairs["prev_lo"], pairs["prev_hi"] = stats.beta.ppf(0.05, a, b), stats.beta.ppf(0.95, a, b)
    pairs["lift"] = pairs.prev / ps
    pairs["pr_lift"] = stats.beta.sf(ps, a, b)                             # Pr(p_os > p_s)
    pairs["visible"] = pairs.skill_uri.isin(visible)
    pairs["_a"], pairs["_b"] = a, b

    # reference levels
    off = E.set_index(["occ_uri", "skill_uri"]).relation
    pairs["relation"] = [off.get((o, s), "") for o, s in zip(pairs.occ_uri, pairs.skill_uri)]
    # first pass: specificity, to know which official links the market confirms
    sup_flag = (pairs.relation != "") & (pairs.pr_lift >= CRED)
    ref = []
    for o in A.occ_uri:
        po = pairs[pairs.occ_uri == o]
        Nn = int(N_o.get(o, 0))
        related = minor_skills.get(str(occ_isco.get(o, ""))[:3], set())
        unrel_vis = [s for s in visible if s not in related]
        prev_unrel = po.set_index("skill_uri").prev_raw.reindex(unrel_vis).fillna(0.0)
        vo = po[po.visible]
        so = po[sup_flag.loc[po.index]]
        all_sup = so.prev_raw.median() if len(so) else np.nan
        r_opt = so[so.relation == "optional"].prev_raw
        r_ess = so[so.relation == "essential"].prev_raw
        ref.append({"occ_uri": o, "N_ads": Nn,
                    "N_null": float(np.quantile(prev_unrel, NULL_Q)) if len(prev_unrel) else np.nan,
                    "R_opt": float(r_opt.median()) if len(r_opt) >= MIN_REF else float(all_sup),
                    "R_ess": float(r_ess.median()) if len(r_ess) >= MIN_REF else float(all_sup),
                    "R_dem": float(all_sup), "n_supported_links": len(so),
                    "R_opt_all": float(vo[vo.relation == "optional"].prev_raw.median()),
                    "R_ess_all": float(vo[vo.relation == "essential"].prev_raw.median()),
                    "n_unrelated_skills": len(unrel_vis)})
    ref = pd.DataFrame(ref).set_index("occ_uri")
    pairs = pairs.join(ref[["N_null", "R_opt", "R_ess", "R_dem"]], on="occ_uri")
    pairs["pr_below_null"] = stats.beta.cdf(pairs.N_null, pairs._a, pairs._b)     # Pr(p_os <= N_o)
    pairs["pr_above_ropt"] = stats.beta.sf(pairs.R_opt, pairs._a, pairs._b)       # Pr(p_os >= R_opt)
    pairs["pr_above_ress"] = stats.beta.sf(pairs.R_ess, pairs._a, pairs._b)       # Pr(p_os >= R_ess)
    pairs["powered"] = stats.binom.cdf(pairs.n, pairs.N, pairs.R_dem.fillna(0)) < ALPHA_POW
    return pairs.drop(columns=["_a", "_b"]), ref, sk, n_all


log(f"Evidence in {WIN_SEL}")
P, REF, SK, NALL = evidence(WIN_SEL)
log(f"Evidence in {WIN_EVAL}")
P1, REF1, _, _ = evidence(WIN_EVAL)


def status(df):
    sup = df.pr_lift >= CRED
    unobs = ~df.visible | ~df.powered
    weak = (df.pr_below_null >= CRED) & (df.pr_lift < 0.5)
    st = np.select([sup, unobs, weak], ["supported", "unobservable", "weak"], default="intermediate")
    return pd.Series(st, index=df.index)


P["status"] = status(P)
P1["status"] = status(P1)
supported_somewhere = set(P.loc[(P.status == "supported"), "skill_uri"])

# ============================================================
# 3. OFFICIAL LINKS
# ============================================================
key = ["occ_uri", "skill_uri"]
ev = E.merge(P.drop(columns=["relation"]), on=key, how="left")
ev.loc[(ev.status == "weak") & ev.apply(
    lambda r: r.skill_uri in set(P.loc[(P.status == "supported") & (P.occ_uri != r.occ_uri), "skill_uri"]),
    axis=1), "status"] = "contradicted"
h1 = P1.set_index(key)[["prev_raw", "lift", "pr_lift", "status"]].add_suffix("_h1")
ev = ev.join(h1, on=key)
ev["same_direction_h1"] = np.select(
    [ev.status == "supported", ev.status.isin(["weak", "contradicted"])],
    [ev.lift_h1 > 1, ev.lift_h1 < 1], default=np.nan)
# preliminary signals for Stage 6 (protections and the implied-skill gate are applied there)
ev["signal_upgrade_to_essential"] = (ev.relation == "optional") & (ev.pr_above_ress >= CRED) & (ev.pr_lift >= CRED)
ev["signal_weak_optional"] = (ev.relation == "optional") & ev.status.isin(["weak", "contradicted"])
ev["signal_weak_essential"] = (ev.relation == "essential") & ev.status.isin(["weak", "contradicted"])

# ============================================================
# 4. ADDITION CANDIDATES (pairs not in ESCO)
# ============================================================
off_pairs = set(zip(E.occ_uri, E.skill_uri))
C = P[[(o, s) not in off_pairs for o, s in zip(P.occ_uri, P.skill_uri)]].copy()
C = C[(C.pr_lift >= CRED) & (C.pr_above_ropt >= CRED)].copy()
C["occupation"] = C.occ_uri.map(dict(zip(A.occ_uri, A.occupation)))
C["skill"] = C.skill_uri.map(skill_label)
C["is_language"] = C.skill_uri.isin(coll["language"])
C["is_transversal"] = C.skill_uri.isin(coll["transversal"])
C["is_boilerplate"] = C.skill_uri.isin(boiler)
C["is_digital"] = C.skill_uri.isin(coll["digital"])
C["in_esco_minor_group"] = [s in minor_skills.get(str(occ_isco.get(o, ""))[:3], set())
                            for o, s in zip(C.occ_uri, C.skill_uri)]
C["candidate_type"] = np.select([C.is_language, C.is_boilerplate, C.is_transversal],
                                ["language", "boilerplate", "transversal (outside profiles by design)"],
                                default="addition candidate")
C["essential_level"] = C.pr_above_ress >= CRED
C = C.join(h1, on=key)
C["same_direction_h1"] = C.lift_h1 > 1

# ============================================================
# 5. E4 SEMANTIC PLAUSIBILITY (optional)
# ============================================================
ev["sem_pct"] = np.nan
C["sem_pct"] = np.nan
if USE_SEMANTIC:
    try:
        from sentence_transformers import SentenceTransformer
        log("Semantic plausibility (E4)")
        model = SentenceTransformer(SBERT_MODEL)
        su = sorted(set(ev.skill_uri) | set(C.skill_uri))
        st_txt = [f"{skill_label.get(s, '')}. {str(skill_desc.get(s, ''))[:300]}" for s in su]
        oc_txt = [f"{lab}. {str(occ_desc.get(u, ''))[:300]}" for u, lab in zip(A.occ_uri, A.occupation)]
        se = model.encode(st_txt, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        oe = model.encode(oc_txt, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        si = {s: i for i, s in enumerate(su)}
        oi = {u: i for i, u in enumerate(A.occ_uri)}
        for df in (ev, C):
            df["sem"] = [float(se[si[s]] @ oe[oi[o]]) for o, s in zip(df.occ_uri, df.skill_uri)]
        base = ev.groupby("occ_uri").sem.apply(np.sort).to_dict()     # o's own ESCO links as reference
        for df in (ev, C):
            df["sem_pct"] = [100.0 * np.searchsorted(base[o], v) / len(base[o]) for o, v in zip(df.occ_uri, df.sem)]
    except Exception as ex:
        log(f"E4 skipped ({ex.__class__.__name__}); install sentence-transformers to compute it")

# ============================================================
# 6. OCCUPATION SUMMARY: references, statuses, digital gap, consistency, coverage
# ============================================================
rows = []
for _, r in A.iterrows():
    o = r.occ_uri
    eo, po = ev[ev.occ_uri == o], P[P.occ_uri == o]
    sko = SK[SK.occ_uri == o]
    st = eo.status.value_counts()
    vis = eo[eo.visible]
    ess, opt = vis[vis.relation == "essential"].prev_raw, vis[vis.relation == "optional"].prev_raw
    auc = (stats.mannwhitneyu(ess, opt, alternative="two-sided").statistic / (len(ess) * len(opt))
           if len(ess) and len(opt) else np.nan)
    spec = po[po.pr_lift >= CRED]
    rows.append({
        "cluster": r.cluster, "occupation": r.occupation, "N_ads_H2": int(REF.loc[o, "N_ads"]),
        "N_ads_H1": int(REF1.loc[o, "N_ads"]) if o in REF1.index else 0,
        "N_null": round(REF.loc[o, "N_null"], 4), "R_opt": round(REF.loc[o, "R_opt"], 4),
        "R_ess": round(REF.loc[o, "R_ess"], 4), "R_dem": round(REF.loc[o, "R_dem"], 4),
        "R_opt_all": round(REF.loc[o, "R_opt_all"], 4), "R_ess_all": round(REF.loc[o, "R_ess_all"], 4),
        "links": len(eo), "share_links_never_mentioned": round((eo.n == 0).mean(), 3),
        **{f"n_{k}": int(st.get(k, 0)) for k in ["supported", "intermediate", "weak", "contradicted", "unobservable"]},
        "ess_beats_opt_AUC": round(auc, 3),
        "demand_coverage": round(sko.skill_uri.isin(set(eo.skill_uri)).mean(), 3) if len(sko) else np.nan,
        "digital_official": round(eo.is_digital.mean(), 3),
        "digital_advertised": round(sko.skill_uri.isin(coll["digital"]).mean(), 3) if len(sko) else np.nan,
        "digital_specific": round(spec.skill_uri.isin(coll["digital"]).mean(), 3) if len(spec) else np.nan,
        "additions": int(((C.occ_uri == o) & (C.candidate_type == "addition candidate")).sum()),
        "additions_essential_level": int(((C.occ_uri == o) & (C.candidate_type == "addition candidate")
                                          & C.essential_level).sum()),
    })
OS = pd.DataFrame(rows).sort_values(["cluster", "occupation"])

# ============================================================
# 7. REPORT
# ============================================================
say()
say("1. Reference levels per occupation (H2 2024)")
say("   N_null = what an unrelated skill reaches; R_opt / R_ess / R_dem = median prevalence of o's SUPPORTED")
say("   optional / essential / all ESCO links; *_all = median over ALL visible links (descriptive)")
with pd.option_context("display.width", 240):
    say(OS[["occupation", "N_ads_H2", "N_ads_H1", "N_null", "R_opt", "R_ess", "R_dem", "R_opt_all", "R_ess_all",
            "share_links_never_mentioned"]].to_string(index=False))
say(f"   ESCO links never mentioned in the occupation's own H2 ads: {(ev.n == 0).mean():.1%}")

say()
say("2. Evidence status of the official links")
tab = pd.crosstab(ev.relation, ev.status, margins=True)
say(tab.to_string())
say("   shares: " + ", ".join(f"{k} {v:.1%}" for k, v in ev.status.value_counts(normalize=True).items()))
with pd.option_context("display.width", 220):
    say(OS[["occupation", "links", "n_supported", "n_intermediate", "n_weak", "n_contradicted",
            "n_unobservable"]].to_string(index=False))

say()
say("3. Do essential links show more demand than optional ones? (AUC: 0.5 = no difference, 1 = always)")
say(OS[["occupation", "ess_beats_opt_AUC", "demand_coverage"]].to_string(index=False))
say(f"   median AUC {OS.ess_beats_opt_AUC.median():.2f}; median demand coverage "
    f"(share of skill mentions in o's ads that ESCO's profile contains) {OS.demand_coverage.median():.1%}")

say()
say("4. Digital gap: share of digital skills in ESCO's profile vs in the ads")
say(OS[["occupation", "digital_official", "digital_advertised", "digital_specific"]].to_string(index=False))

say()
say(f"5. Direction check on {WIN_EVAL} (same direction of lift: > 1 for supported, < 1 for weak)")
for s_ in ["supported", "weak"]:
    m = ev.status == s_ if s_ == "supported" else ev.status.isin(["weak", "contradicted"])
    say(f"   {s_:10s} in H2: {m.sum():5,} links; same in H1: {ev.loc[m, 'same_direction_h1'].astype(float).mean():.1%}")

say()
say("6. Weak or contradicted links (examples, lowest lift first)")
wk = ev[ev.status.isin(["weak", "contradicted"])].sort_values("lift")
say(wk[["occupation", "skill", "relation", "status", "prev_raw", "lift", "protected_any"]]
    .head(25).round(4).to_string(index=False))
say(f"   weak/contradicted: {len(wk):,} links, of which optional and unprotected "
    f"{((wk.relation == 'optional') & ~wk.protected_any).sum():,}; essential {(wk.relation == 'essential').sum():,}")

say()
say("7. Addition candidates (pairs not in ESCO, specific AND at least as common as o's optional links)")
say("   by type: " + ", ".join(f"{k} {v}" for k, v in C.candidate_type.value_counts().items()))
ac = C[C.candidate_type == "addition candidate"].sort_values("lift", ascending=False)
for occ_, g in ac.groupby("occupation"):
    say(f"  {occ_}: " + " | ".join(f"{s} ({p:.0%}, lift {l:.1f})" for s, p, l in
                                     zip(g.skill.head(6), g.prev_raw.head(6), g.lift.head(6))))
say(f"   same direction in H1 2025: {ac.same_direction_h1.mean():.1%} of addition candidates")
tv = C[C.candidate_type != "addition candidate"]
if len(tv):
    say("   language / transversal / boilerplate skills demanded but outside profiles (most frequent): "
        + ", ".join(tv.groupby("skill").size().sort_values(ascending=False).head(10).index))

say()
say(f"8. Upgrade signals (optional link at least as common as o's essential links, and specific): "
    f"{int(ev.signal_upgrade_to_essential.sum())}")
say(ev[ev.signal_upgrade_to_essential][["occupation", "skill", "prev_raw", "lift"]]
    .sort_values("prev_raw", ascending=False).head(15).round(3).to_string(index=False))
if ev.sem_pct.notna().any():
    say()
    say(f"9. E4 semantic percentile: median for weak links {ev.loc[ev.status == 'weak', 'sem_pct'].median():.0f}, "
        f"supported {ev.loc[ev.status == 'supported', 'sem_pct'].median():.0f}; "
        f"addition candidates {C.sem_pct.median():.0f}")

# ============================================================
# 8. SAVE AND FIGURES
# ============================================================
OUT.mkdir(parents=True, exist_ok=True)
ev.to_parquet(OUT / "edge_evidence.parquet", index=False)
C.to_parquet(OUT / "addition_candidates.parquet", index=False)
P[["occ_uri", "skill_uri", "n", "N", "p_s", "prev_raw", "lift", "pr_lift", "visible", "relation"]].to_parquet(
    OUT / "pair_evidence.parquet", index=False)                  # all observed pairs, H2 (read by Stage 5)
OS.to_csv(OUT / "occupation_evidence.csv", index=False)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    # F4 evidence status per occupation
    order = ["supported", "intermediate", "unobservable", "weak", "contradicted"]
    cols = {"supported": "#2a78d6", "intermediate": "#c3c2b7", "unobservable": "#e6e5e0",
            "weak": "#eb6834", "contradicted": "#e34948"}
    sh = pd.crosstab(ev.occupation, ev.status, normalize="index").reindex(columns=order, fill_value=0)
    sh = sh.loc[sh.supported.sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(sh) + 1.2))
    left = np.zeros(len(sh))
    for k in order:
        ax.barh(sh.index, sh[k] * 100, left=left, color=cols[k], label=k, edgecolor="white", linewidth=0.8)
        left += sh[k].values * 100
    ax.set_xlabel("% of the occupation's ESCO links")
    ax.set_title("Evidence status of ESCO links in H2 2024 job ads", loc="left", fontsize=9, color=INK)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=7)
    fig.savefig(OUT / "F4_evidence_status.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # F3 digital gap
    d = OS.sort_values("digital_specific")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.hlines(y, d[["digital_official", "digital_specific"]].min(axis=1) * 100,
              d[["digital_official", "digital_specific"]].max(axis=1) * 100, color=GRID, linewidth=2)
    ax.scatter(d.digital_official * 100, y, color="#2a78d6", s=30, label="ESCO profile", zorder=3)
    ax.scatter(d.digital_specific * 100, y, color="#eb6834", s=30, label="skills specific to the ads", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.occupation)
    ax.set_xlabel("% digital skills")
    ax.set_title("Digital share: ESCO profile vs occupation-specific skills in ads", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "F3_digital_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("Figures F3, F4 saved")
except Exception as ex:
    log(f"Figures skipped: {ex}")

say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "stage4_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
