"""
ZenESCO - Stages 5-6: implied-skill gate and candidate screening
================================================================

Objective
    Turn the Stage 4 evidence into a SHORT, defensible list of candidate edits, and show why
    frequency-based pruning would be wrong. Nothing is changed in ESCO yet: Stage 7 tests every
    candidate on held-out ads before anything is recommended.

Stage 5 - implied-skill gate (is a weak link really weak, or just not written in ads?)
    G1 Advertising discount. Some skills are rarely written even when required: in ads, general
       skills are mentioned less than specific ones. For ESCO links of the analysed occupations,
       the share mentioned at least once is computed per generality quintile; delta_g = that share
       divided by the best quintile's share. The power test of Stage 4 is repeated with the expected
       level R_dem(o) * delta_g: a weak link stays weak only if its absence is still significant.
    G2 Implication through related skills. A weak link (o, s) is IMPLIED if ESCO relates s to a skill
       r (broader/narrower in the skills pillar, or the skill-to-skill relations file) and r is
       supported in o's ads: the demand is there, written at another level of detail.

Stage 6 - screening (first matching rule, per official link)
    1 protected (regulated, safety/legal, transversal, last link)      -> retain: protected
    2 unobservable, or absence not significant after the discount      -> retain: no evidence
    3 implied by a supported related skill                             -> retain: implied
    4 weak or contradicted, optional                                   -> DELETE candidate
    5 weak or contradicted, essential                                  -> DEMOTE candidate (upgrade -> optional)
    6 optional, specific and as common as o's essential links;
      not boilerplate / language / transversal                         -> UPGRADE candidate (-> essential)
    7 otherwise                                                         -> retain: supported / adequate
    Pairs not in ESCO: Stage 4 addition candidates that are not language / transversal / boilerplate
                                                                        -> ADD candidate
    E4 (if computed): candidates whose semantic percentile is below 25 are flagged low-plausibility
    (they go to expert review in Stage 9 unless the utility test is clearly positive).

Baseline for comparison (RQ1): frequency pruning = delete every ESCO link never mentioned in the
occupation's ads. The report shows how many of those ZenESCO keeps, and why.

Inputs   BASE\\stage4\\edge_evidence.parquet, addition_candidates.parquet, pair_evidence.parquet,
         occupation_evidence.csv; BASE\\stage1\\boilerplate_candidates.csv;
         BASE\\esco\\ broaderRelationsSkillPillar_en.csv, skillSkillRelations_en.csv
Outputs  BASE\\stage6\\screened_links.parquet     every official link with gate results and screening action
         BASE\\stage6\\add_candidates.parquet     screened additions
         BASE\\stage6\\table3_screening.csv       per occupation: counts per action (Table 3)
         BASE\\stage6\\F6_advertising_discount.png, stage5_6_report.txt
Run      python stage5_6_screening.py           # BASE = C:\\skillab_data
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
S1, S4, ESCO = BASE / "stage1", BASE / "stage4", BASE / "esco"
OUT = BASE / "stage6"
CRED = 0.95
ALPHA_POW = 0.05
N_GEN_GROUPS = 5
SEM_LOW = 25
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


ev = pd.read_parquet(S4 / "edge_evidence.parquet")
C = pd.read_parquet(S4 / "addition_candidates.parquet")
PE = pd.read_parquet(S4 / "pair_evidence.parquet")
OS = pd.read_csv(S4 / "occupation_evidence.csv")
bp_f = S1 / "boilerplate_candidates.csv"
boiler = set(pd.read_csv(bp_f).query("candidate == True").skill_uri) if bp_f.exists() else set()
R_dem = ev.groupby("occ_uri").R_dem.first()

say("ZenESCO - Stages 5-6 report: implied-skill gate and candidate screening")
say("=" * 74)
say(f"Official links: {len(ev):,} in {ev.occ_uri.nunique()} occupations")

# ============================================================
# STAGE 5 - G1 advertising discount by skill generality
# ============================================================
vis = ev[ev.visible].copy()
vis["gen_group"] = pd.qcut(vis.generality.rank(method="first"), N_GEN_GROUPS, labels=False) + 1
edges_q = np.quantile(vis.generality, np.linspace(0, 1, N_GEN_GROUPS + 1))
mention = vis.groupby("gen_group").apply(lambda g: (g.n > 0).mean())
delta = mention / mention.max()
gen_range = vis.groupby("gen_group").generality.agg(["min", "max"])
ev["gen_group"] = np.clip(np.searchsorted(edges_q[1:-1], ev.generality, side="right") + 1, 1, N_GEN_GROUPS)
ev["ad_discount"] = ev.gen_group.map(delta).fillna(1.0)
ev["expected_level"] = ev.occ_uri.map(R_dem) * ev.ad_discount
ev["powered_adj"] = stats.binom.cdf(ev.n, ev.N, ev.expected_level.fillna(0)) < ALPHA_POW

say()
say("Stage 5 - G1 advertising discount: share of ESCO links mentioned at least once, by skill generality")
t = pd.DataFrame({"generality (ESCO occupations using the skill)":
                  [f"{int(a)}-{int(b)}" for a, b in zip(gen_range["min"], gen_range["max"])],
                  "links": vis.groupby("gen_group").size().values,
                  "share mentioned": mention.round(3).values, "discount delta": delta.round(2).values})
say(t.to_string(index=False))

# ============================================================
# STAGE 5 - G2 implication through related skills
# ============================================================
related = defaultdict(set)
b = pd.read_csv(esco_file("broaderRelationsSkillPillar_en.csv"), dtype=str)
b = b[(b.conceptType == "KnowledgeSkillCompetence") & (b.broaderType == "KnowledgeSkillCompetence")]
for x, y in zip(b.conceptUri, b.broaderUri):
    related[x].add(y)
    related[y].add(x)
r = pd.read_csv(esco_file("skillSkillRelations_en.csv"), dtype=str)
for x, y in zip(r.originalSkillUri, r.relatedSkillUri):
    related[x].add(y)
    related[y].add(x)
supported = PE[PE.pr_lift >= CRED].groupby("occ_uri").skill_uri.agg(set).to_dict()


def implied_by(o, s):
    hit = related.get(s, set()) & supported.get(o, set())
    return sorted(hit)[:3]


weakish = ev.status.isin(["weak", "contradicted"])
ev["implied_by"] = [implied_by(o, s) if w else [] for o, s, w in zip(ev.occ_uri, ev.skill_uri, weakish)]
ev["implied"] = ev.implied_by.map(len) > 0
say()
say(f"Stage 5 - G2: weak/contradicted links implied by a supported related skill: "
    f"{int((weakish & ev.implied).sum())} of {int(weakish.sum())} "
    f"(skills with any ESCO relation: {sum(1 for s in ev.skill_uri.unique() if related.get(s))} "
    f"of {ev.skill_uri.nunique()})")

# ============================================================
# STAGE 6 - screening of official links
# ============================================================
not_adv = ev.skill_uri.isin(boiler) | ev.is_language | ev.is_transversal
conds = [ev.protected_any,
         (ev.status == "unobservable") | (weakish & ~ev.powered_adj),
         weakish & ev.implied,
         weakish & (ev.relation == "optional"),
         weakish & (ev.relation == "essential"),
         ev.signal_upgrade_to_essential & ~not_adv]
acts = ["retain: protected", "retain: no evidence", "retain: implied", "DELETE candidate",
        "DEMOTE candidate", "UPGRADE candidate"]
ev["action"] = np.select(conds, acts, default="retain: supported/adequate")
ev.loc[(ev.status == "unobservable") & ev.protected_any, "action"] = "retain: protected"
ev["discount_saved"] = weakish & ev.powered & ~ev.powered_adj & ~ev.protected_any
if "sem_pct" in ev.columns and ev.sem_pct.notna().any():
    ev["low_plausibility"] = ev.sem_pct < SEM_LOW
else:
    ev["low_plausibility"] = np.nan

A = C[C.candidate_type == "addition candidate"].copy()
A["action"] = "ADD candidate"
A["proposed_type"] = np.where(A.essential_level, "essential", "optional")
A["low_plausibility"] = (A.sem_pct < SEM_LOW) if A.sem_pct.notna().any() else np.nan

say()
say("Stage 6 - screening of the official links")
vc = ev.action.value_counts()
for a in acts + ["retain: supported/adequate"]:
    say(f"  {a:28s} {vc.get(a, 0):5,} ({vc.get(a, 0) / len(ev):6.1%})")
say(f"  weak links kept only because of the advertising discount: {int(ev.discount_saved.sum())}")
say(f"  ADD candidates (pairs not in ESCO): {len(A):,} "
    f"(essential level {int(A.essential_level.sum())}, optional level {int((~A.essential_level).sum())})")
if ev.low_plausibility.notna().any():
    cand = ev.action.isin(["DELETE candidate", "DEMOTE candidate", "UPGRADE candidate"])
    say(f"  low semantic plausibility (E4 < {SEM_LOW}th pct): {int(ev.loc[cand, 'low_plausibility'].sum())} of "
        f"{int(cand.sum())} edit candidates; {int(A.low_plausibility.sum())} of {len(A)} additions")
else:
    say("  E4 semantic plausibility not available (rerun Stage 4 with sentence-transformers installed)")

# ============================================================
# RQ1 baseline: frequency pruning
# ============================================================
naive = ev.n == 0
say()
say("Comparison with frequency pruning (delete every ESCO link never mentioned in the occupation's ads)")
say(f"  frequency pruning would delete {int(naive.sum()):,} links ({naive.mean():.1%}), "
    f"of which essential {int((naive & (ev.relation == 'essential')).sum())}")
nb = ev[naive].action.value_counts()
for a, n in nb.items():
    say(f"    ZenESCO: {a:28s} {n:5,} ({n / naive.sum():.1%})")
zen = ev.action.isin(["DELETE candidate", "DEMOTE candidate"])
say(f"  ZenESCO delete/demote candidates: {int(zen.sum())}; never mentioned: {int((zen & naive).sum())}; "
    f"mentioned but weak: {int((zen & ~naive).sum())}")

# ============================================================
# TABLE 3 and examples
# ============================================================
tab = pd.crosstab(ev.occupation, ev.action)
tab["ADD candidate"] = A.groupby("occupation").size().reindex(tab.index).fillna(0).astype(int)
tab["links"] = ev.groupby("occupation").size()
tab["never_mentioned"] = ev.groupby("occupation").n.apply(lambda x: int((x == 0).sum()))
order = ["links", "never_mentioned", "retain: protected", "retain: no evidence", "retain: implied",
         "retain: supported/adequate", "DELETE candidate", "DEMOTE candidate", "UPGRADE candidate", "ADD candidate"]
tab = tab.reindex(columns=[c for c in order if c in tab.columns], fill_value=0)
say()
say("Table 3 - screening per occupation")
with pd.option_context("display.width", 260):
    say(tab.to_string())

say()
for act in ["DELETE candidate", "DEMOTE candidate", "UPGRADE candidate"]:
    g = ev[ev.action == act]
    if not len(g):
        continue
    say(f"{act}s, examples (per occupation, up to 3)")
    srt = g.sort_values("prev_raw", ascending=(act != "UPGRADE candidate"))
    for o, gg in srt.groupby("occupation"):
        say(f"  {o}: " + " | ".join(f"{s} ({p:.1%})" for s, p in zip(gg.skill.head(3), gg.prev_raw.head(3))))
say("ADD candidates, top 4 per occupation by share of ads")
for o, gg in A.sort_values("prev_raw", ascending=False).groupby("occupation"):
    say(f"  {o}: " + " | ".join(f"{s} ({p:.0%}, {t})" for s, p, t in
                                 zip(gg.skill.head(4), gg.prev_raw.head(4), gg.proposed_type.head(4))))

# ============================================================
# SAVE
# ============================================================
OUT.mkdir(parents=True, exist_ok=True)
ev.assign(implied_by=ev.implied_by.map(lambda x: "|".join(x))).to_parquet(OUT / "screened_links.parquet", index=False)
A.to_parquet(OUT / "add_candidates.parquet", index=False)
tab.to_csv(OUT / "table3_screening.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(t.iloc[:, 0], t["share mentioned"] * 100, color="#2a78d6", width=0.6)
    for i, v in enumerate(t["share mentioned"]):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=7, color="#52514e")
    ax.set_xlabel("skill generality: ESCO occupations using the skill (quintiles)")
    ax.set_ylabel("% of ESCO links mentioned in ads")
    ax.set_title("Advertising discount: general skills are written less often", loc="left", fontsize=9)
    fig.savefig(OUT / "F6_advertising_discount.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
except Exception as ex:
    log(f"Figure skipped: {ex}")
say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "stage5_6_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
