"""
ZenESCO - three "change" figures in the style of the reference figures (run after Stage 7)
=========================================================================================

All three use the edits that pass the Stage 7 held-out test (deletions/demotions non-inferior;
additions/upgrades with a 90% interval above 0). Final recommendations come in Stage 9.

F9a  How much closer job ads get to their occupation's ESCO profile (reference: "skill vector change" boxplots)
     Unit = held-out job ad (H1 2025) of an analysed occupation.
     Change = cos(ad skills, UPDATED profile) - cos(ad skills, ORIGINAL profile); IDF-weighted, essential 1,
     optional 0.5 (same scoring as Stage 7). > 0: the updated ESCO describes the ad better.
       a  by number of skills in the ad          b  by seniority stated in the title
       c  by occupation level (ISCO major group)  d  by occupation, sorted by median
F9b  Specific, intermediate and general skills (reference: skill-level distributions by skill group)
     Skill groups = terciles of ESCO generality (number of ESCO occupations using the skill).
     Demand level of a skill in an occupation = share of the occupation's H2 2024 ads mentioning it,
     binned into levels 0-7 (0 = never, 1 = up to 0.5%, 2 = 0.5-1%, 3 = 1-2%, 4 = 2-5%, 5 = 5-10%,
     6 = 10-20%, 7 = over 20%). Boxes: across the skills of the group, how many of the analysed occupations
     demand the skill at each level. Small histograms: two example skills per group (one added by ZenESCO).
     Panel d: nestedness with real numbers on the UPDATED ESCO graph (all ESCO occupations; the analysed
     ones with ZenESCO's edits): P(A|B) = share of occupations with skill B that also have A.
F9c  Demand over time (reference: stacked area + stacked bars by month)
     Left: skill mentions per ad, per month, in the analysed occupations' ads, by how the updated ESCO
     treats them. Right: mentions per 100 ads of the skills ZenESCO edits, by edit type.
     Hatched = held-out months (H1 2025), never used to choose the edits.

Run   python figures_change.py            # BASE = C:\\skillab_data       outputs in BASE\\figures
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy import sparse

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ESCO = BASE / "esco"
ALPHA = 0.5

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#008300", "#6250d6", "#e34948", "#e87ba4")
SPEC, INTER, GEN = "#6f7fd1", "#8a8984", "#e34948"          # specific / intermediate / general (reference colours)
LEVEL_COL = {"Managers (ISCO 1)": VIOLET, "Professionals (ISCO 2)": BLUE, "Technicians & assoc. prof. (ISCO 3)": AQUA}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=SURF)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {name}")


def letter(ax, l, x=-0.14, y=1.04):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")


def esco_file(name):
    hits = [p for p in ESCO.glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    return hits[0] if hits else None


def short(t, n=28):
    t = str(t)
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


# =====================================================================================
# DATA
# =====================================================================================
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
ADD = pd.read_parquet(BASE / "stage6" / "add_candidates.parquet")
EF = pd.read_parquet(BASE / "stage7" / "edit_effects.parquet")
OCC = sorted(EV.occ_uri.unique())
olab = EV.groupby("occ_uri").occupation.first().to_dict()
passed = {(o, s_, a_): (bool(i) if a_ in ("ADD candidate", "UPGRADE candidate") else bool(n))
          for o, s_, a_, i, n in zip(EF.occupation, EF.skill_uri, EF.action, EF.improves, EF.noninferior)}

# use the FINAL Stage 9 decisions when they exist (otherwise the Stage 7 pass/fail)
_r1 = BASE / "stage9" / "R1_all_links.csv"
if _r1.exists():
    _R = pd.read_csv(_r1)
    _amap = {"Delete": "DELETE candidate", "Upgrade -> optional": "DEMOTE candidate",
             "Upgrade -> essential": "UPGRADE candidate", "Add (essential)": "ADD candidate", "Add (optional)": "ADD candidate"}
    passed = {k: False for k in passed}
    for o_, s__, rec_ in zip(_R.occupation, _R.skill_uri, _R.recommendation):
        if rec_ in _amap:
            passed[(o_, s__, _amap[rec_])] = True
    print("Using final Stage 9 decisions (R1_all_links.csv)")

ORIG, UPD, EDIT = {}, {}, {}           # profiles (skill -> weight) and edit type per (occ, skill)
for o in OCC:
    e = EV[EV.occ_uri == o]
    ORIG[o] = {s_: (1.0 if r == "essential" else ALPHA) for s_, r in zip(e.skill_uri, e.relation)}
    UPD[o] = dict(ORIG[o])
    for s_, a_ in zip(e.skill_uri, e.action):
        if not passed.get((olab[o], s_, a_), False):
            continue
        if a_ == "DELETE candidate":
            UPD[o].pop(s_, None)
            EDIT[(o, s_)] = "deleted"
        elif a_ == "DEMOTE candidate":
            UPD[o][s_] = ALPHA
            EDIT[(o, s_)] = "demoted"
        elif a_ == "UPGRADE candidate":
            UPD[o][s_] = 1.0
            EDIT[(o, s_)] = "upgraded"
    a = ADD[ADD.occ_uri == o]
    for s_, t in zip(a.skill_uri, a.proposed_type):
        if passed.get((olab[o], s_, "ADD candidate"), False):
            UPD[o][s_] = 1.0 if t == "essential" else ALPHA
            EDIT[(o, s_)] = "added"

ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet")
ads = ads[ads.analysable & ads.window.isin(["H2_2024", "H1_2025"])].copy()
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
ads = ads.merge(occ_of, on="ad_id")
ads = ads[ads.occ_uri.isin(set(OCC))].reset_index(drop=True)
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(ads.ad_id))].drop_duplicates().merge(ads[["ad_id", "occ_uri", "window"]], on="ad_id")

rel = pd.read_csv(esco_file("occupationSkillRelations_en.csv"), dtype=str, usecols=["occupationUri", "skillUri"])
N_OCC = rel.occupationUri.nunique()
gen = rel.groupby("skillUri").occupationUri.nunique()
occs = pd.read_csv(esco_file("occupations_en.csv"), dtype=str, usecols=["conceptUri", "iscoGroup", "preferredLabel"])
isco = dict(zip(occs.conceptUri, occs.iscoGroup))
skills = pd.read_csv(esco_file("skills_en.csv"), dtype=str, keep_default_na=False,
                     usecols=["conceptUri", "preferredLabel", "reuseLevel"])
slabel = dict(zip(skills.conceptUri, skills.preferredLabel))
coll = set()
for f in ["languageSkillsCollection_en.csv", "transversalSkillsCollection_en.csv"]:
    p = esco_file(f)
    if p is not None:
        coll |= set(pd.read_csv(p, dtype=str).conceptUri)
coll |= set(skills.conceptUri[skills.reuseLevel == "transversal"])


_bp = BASE / "stage1" / "boilerplate_candidates.csv"
BOILER = set(pd.read_csv(_bp).query("candidate == True").skill_uri) if _bp.exists() else set()


def idf(s_):
    return max(np.log(N_OCC / gen.get(s_, 1)), 0.05)


# =====================================================================================
# F9a  per-ad change in ad-profile similarity
# =====================================================================================
def f9a():
    test = ads[ads.window == "H1_2025"].copy()
    sk = ask[ask.window == "H1_2025"].groupby("ad_id").skill_uri.agg(list)
    test["skills"] = test.ad_id.map(sk)
    test = test[test.skills.notna()].copy()

    def cos(skl, prof):
        num = sum(prof.get(s_, 0) * idf(s_) for s_ in skl)
        den = np.sqrt(sum((w * idf(s_)) ** 2 for s_, w in prof.items())) * np.sqrt(len(skl))
        return num / den if den else 0.0

    test["before"] = [cos(s_, ORIG[o]) for s_, o in zip(test.skills, test.occ_uri)]
    test["after"] = [cos(s_, UPD[o]) for s_, o in zip(test.skills, test.occ_uri)]
    test["change"] = test.after - test.before
    test["n_sk"] = test.skills.map(len)
    test["occupation"] = test.occ_uri.map(olab)
    lv = {"1": "Managers (ISCO 1)", "2": "Professionals (ISCO 2)", "3": "Technicians & assoc. prof. (ISCO 3)"}
    test["level"] = test.occ_uri.map(lambda u: lv.get(str(isco.get(u, ""))[:1], "Other"))
    title = test.get("title_raw", pd.Series("", index=test.index)).fillna("").str.lower()
    sen = re.compile(r"\b(?:senior|sr\.?|erfaren|experienced|expérimenté|confirmé|expert|principal|lead|"
                     r"chief|head|staff|ledande|seniora?)\b")
    jun = re.compile(r"\b(?:junior|jr\.?|entry|graduate|nyexad|débutant|associate|assistant|trainee|junior)\b")
    test["seniority"] = np.where(title.str.contains(sen), "Senior / lead",
                                 np.where(title.str.contains(jun), "Junior / entry", "Not stated"))
    if "experience_level" in test.columns and test.experience_level.notna().mean() > 0.5:
        test["seniority"] = test.experience_level.fillna("Not stated")

    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.9], hspace=0.5, wspace=0.3)
    y_lab = "Change in ad–profile\nsimilarity"

    def boxes(ax, groups, data, cols, xlabels, rot=0):
        bp = ax.boxplot([data[g] for g in groups], notch=True, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=INK2, lw=0.7),
                        capprops=dict(color=INK2, lw=0.7), boxprops=dict(lw=0.7, edgecolor=INK2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(xlabels, rotation=rot, fontsize=6.6)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    # (a) number of skills in the ad
    ax = fig.add_subplot(gs[0, 0])
    bins = [0, 3, 6, 9, 12, 15, 20, 30, 1000]
    names = ["1–3", "4–6", "7–9", "10–12", "13–15", "16–20", "21–30", "31+"]
    test["nb"] = pd.cut(test.n_sk, bins, labels=names)
    g = [n for n in names if (test.nb == n).sum() >= 10]
    boxes(ax, g, {n: test.loc[test.nb == n, "change"] for n in g}, [plt.cm.Reds(0.35 + 0.08 * i) for i in range(len(g))], g, 45)
    ax.set_ylabel(y_lab, color=RED)
    ax.set_xlabel("Skills listed in the ad")
    ax.set_title("Skill richness of ads", fontsize=8, fontweight="bold")
    letter(ax, "a", -0.3)
    # (b) seniority
    ax = fig.add_subplot(gs[0, 1])
    g = [n for n in ["Junior / entry", "Not stated", "Senior / lead"] if (test.seniority == n).sum() >= 10]
    if not g:
        g = list(test.seniority.value_counts().index[:4])
    boxes(ax, g, {n: test.loc[test.seniority == n, "change"] for n in g}, [ORANGE] * len(g),
          [n.replace(" / ", " /\n") for n in g])
    ax.set_title("Seniority of jobs", fontsize=8, fontweight="bold")
    letter(ax, "b", -0.3)
    # (c) occupation level
    ax = fig.add_subplot(gs[0, 2])
    g = [n for n in LEVEL_COL if (test.level == n).sum() >= 10]
    boxes(ax, g, {n: test.loc[test.level == n, "change"] for n in g}, [LEVEL_COL[n] for n in g],
          [{"Managers (ISCO 1)": "Managers\n(ISCO 1)", "Professionals (ISCO 2)": "Profes-\nsionals\n(ISCO 2)",
            "Technicians & assoc. prof. (ISCO 3)": "Technicians\n& assoc. prof.\n(ISCO 3)"}[n] for n in g])
    ax.set_title("Occupation level of jobs", fontsize=8, fontweight="bold")
    letter(ax, "c", -0.3)
    # (d) by occupation
    ax = fig.add_subplot(gs[1, :])
    med = test.groupby("occupation").change.median().sort_values(ascending=False)
    g = list(med.index)
    lvl = test.groupby("occupation")["level"].first()
    boxes(ax, g, {n: test.loc[test.occupation == n, "change"] for n in g}, [LEVEL_COL.get(lvl[n], MUTED) for n in g], [""] * len(g))
    whisks, labs = [], []
    for n in g:
        v = test.loc[test.occupation == n, "change"]
        q1_, q3_ = np.percentile(v, [25, 75])
        whisks.append(min(v.max(), q3_ + 1.5 * (q3_ - q1_)))
        labs.append(f"{n}  (n={len(v)})")
    lows = []
    for n in g:
        v = test.loc[test.occupation == n, "change"]
        q1_, q3_ = np.percentile(v, [25, 75])
        lows.append(max(v.min(), q1_ - 1.5 * (q3_ - q1_)))
    ymin = min(min(lows), 0) - 0.02 * (max(whisks) - min(lows))
    ax.set_ylim(ymin, max(whisks) * 1.05)
    txts = [ax.text(i + 1, w, t, rotation=90, ha="center", va="bottom", fontsize=6.4,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6)) for i, (w, t) in enumerate(zip(whisks, labs))]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_h = ax.get_window_extent(rend).height
    fr = [tt.get_window_extent(rend).height / ax_h for tt in txts]   # label height as a share of the axes
    for tt in txts:
        tt.remove()
    ytop = ymin + max((w - ymin) / max(0.96 - f, 0.25) for w, f in zip(whisks, fr))
    ax.set_ylim(ymin, ytop)
    for i, (w, t) in enumerate(zip(whisks, labs)):
        ax.text(i + 1, w + 0.01 * (ytop - ymin), t, rotation=90, ha="center", va="bottom", fontsize=6.4,
                color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6))
    ax.set_ylabel("Change in similarity between each held-out ad\nand its occupation's ESCO profile")
    ax.set_xlabel("Analysed occupation (sorted by median change)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=c, ms=6, label=k) for k, c in LEVEL_COL.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=6.4)
    letter(ax, "d", -0.07, 1.01)
    share_pos = (test.change > 0).mean()
    ax.set_title(f"Held-out ads (H1 2025): {len(test):,}  ·  closer to their occupation's profile: {share_pos:.0%}  ·  "
                 f"median change {test.change.median():+.3f}", fontsize=7, color=INK2, loc="right")
    save(fig, "F9a_profile_change_boxplots")
    return test


# =====================================================================================
# F9b  specific / intermediate / general skill groups + nestedness schematic
# =====================================================================================
def f9b():
    h2 = ask[ask.window == "H2_2024"]
    n_o = ads[ads.window == "H2_2024"].groupby("occ_uri").size()
    prev = (h2.groupby(["occ_uri", "skill_uri"]).ad_id.nunique() / h2.groupby(["occ_uri", "skill_uri"]).occ_uri.first().map(n_o))
    P = prev.unstack(0).reindex(columns=OCC).fillna(0.0)                 # skills x occupations
    P = P[P.index.isin(slabel.keys())]
    edges = [0, 1e-12, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.01]
    L = P.apply(lambda c: pd.cut(c, edges, labels=False, right=False)).fillna(0).astype(int)
    reuse = dict(zip(skills.conceptUri, skills.reuseLevel))
    rmap = {"occupation-specific": "specific", "sector-specific": "specific", "cross-sector": "intermediate",
            "transversal": "general"}
    grp = pd.Series([("general" if s_ in coll else rmap.get(reuse.get(s_, ""), "intermediate")) for s_ in P.index],
                    index=P.index)
    added_sk = {s_ for (o, s_), t in EDIT.items() if t == "added"}
    fig = plt.figure(figsize=(7.6, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.32, 1, 0.95], hspace=0.28, wspace=0.28)
    meta = [("specific", SPEC, "Specific skill group\n(ESCO: occupation- or sector-specific)"),
            ("intermediate", INTER, "Intermediate skill group\n(ESCO: cross-sector)"),
            ("general", GEN, "General skill group\n(ESCO: transversal & language)")]
    for j, (gname, col, title) in enumerate(meta):
        sk = P.index[grp == gname]
        counts = np.array([[(L.loc[s_] == lv).sum() for lv in range(8)] for s_ in sk]).reshape(-1, 8)  # skills x levels
        ax = fig.add_subplot(gs[1, j])
        bp = ax.boxplot([counts[:, lv] for lv in range(8)], positions=range(8), widths=0.6, patch_artist=True,
                        showfliers=True, flierprops=dict(marker="o", ms=2, mfc=col, mec="none", alpha=0.4),
                        medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=col, lw=0.8),
                        capprops=dict(color=col, lw=0.8), boxprops=dict(lw=0.6, edgecolor=col))
        for b in bp["boxes"]:
            b.set_facecolor(col)
            b.set_alpha(0.35)
        ax.set_xlabel("Demand level in the occupation's ads")
        if j == 0:
            ax.set_ylabel(f"Number of analysed occupations (of {len(OCC)})")
        ax.set_ylim(-0.5, len(OCC) + 0.5)
        ax.set_title(title, fontsize=7.6, fontweight="bold", color=col)
        ax.text(0.98, 0.97, f"{len(sk)} skills", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
        letter(ax, "abc"[j], -0.16, 1.62)
        # two example skills: one added by ZenESCO (if any), one most demanded
        tot = P.loc[sk].sum(axis=1).sort_values(ascending=False)
        tot = tot[~tot.index.isin(BOILER)]
        ex = [s_ for s_ in tot.index if s_ in added_sk][:1] + [s_ for s_ in tot.index if s_ not in added_sk][:1]
        ex = list(dict.fromkeys(ex + list(tot.index[:3])))[:2]
        sub = gs[0, j].subgridspec(1, 2, wspace=0.15)
        for k, s_ in enumerate(ex):
            hx = fig.add_subplot(sub[0, k])
            hx.bar(range(8), [(L.loc[s_] == lv).sum() for lv in range(8)], color=col, alpha=0.8, width=0.8)
            hx.set_xticks([])
            hx.set_yticks([])
            hx.spines["left"].set_visible(False)
            hx.set_title(short(slabel.get(s_, ""), 17) + ("\n(added by ZenESCO)" if s_ in added_sk else "\n "),
                         fontsize=5.6, color=GREEN if s_ in added_sk else INK2, pad=2, linespacing=1.0)
    fig.text(0.5, 0.385, "Demand level: 0 = never mentioned, 1 ≤ 0.5%, 2 ≤ 1%, 3 ≤ 2%, 4 ≤ 5%, 5 ≤ 10%, 6 ≤ 20%, 7 > 20% "
             "of the occupation's ads (H2 2024)", ha="center", fontsize=6.2, color=MUTED)
    fig.text(0.13, 0.965, "High specificity", fontsize=7, color=INK2)
    fig.text(0.87, 0.965, "Low specificity", fontsize=7, color=INK2, ha="right")
    fig.add_artist(FancyArrowPatch((0.24, 0.968), (0.76, 0.968), transform=fig.transFigure, arrowstyle="<|-|>",
                                   mutation_scale=8, color=INK2, lw=0.8))

    # ---- (d) nestedness on the UPDATED ESCO graph: real conditional probabilities
    occ_sk = rel.groupby("occupationUri").skillUri.agg(set).to_dict()
    for o in OCC:
        occ_sk[o] = set(UPD[o])
    has = {}
    for o_, ss in occ_sk.items():
        for s_ in ss:
            has.setdefault(s_, set()).add(o_)

    def cond(a, b):                       # P(a | b), as counts
        ob = has.get(b, set())
        return len(ob & has.get(a, set())), len(ob)

    gmap = dict(zip(P.index, grp))
    demand = P.sum(axis=1)
    ok_sk = [s_ for s_ in demand.sort_values(ascending=False).index
             if s_ not in coll and s_ not in BOILER and len(has.get(s_, ())) >= 4]
    top = ok_sk[:60]
    pairs, used = [], set()

    def best_partner(a_, pool, min_ab=3):
        out = None
        for b_ in pool:
            if b_ == a_ or gen.get(b_, 0) <= 2 * gen.get(a_, 0):
                continue
            n_ab, n_b = cond(a_, b_)
            n_a = len(has.get(a_, ()))
            if n_ab >= min_ab:
                sc = (n_ab / n_a, n_ab)
                if out is None or sc > out[0]:
                    out = (sc, b_)
        return out
    # pair 1: the most demanded skill ZenESCO ADDS, nested in a more general skill
    for a_ in [s_ for s_ in ok_sk if s_ in added_sk][:15]:
        r_ = best_partner(a_, top)
        if r_ and r_[0][0] >= 0.5:
            pairs.append(("big", a_, r_[1]))
            used |= {a_, r_[1]}
            break
    # pair 2: another nested pair among the most demanded skills
    for a_ in top:
        if a_ in used:
            continue
        r_ = best_partner(a_, top)
        if r_ and r_[0][0] >= 0.6 and r_[1] not in used:
            pairs.append(("p2", a_, r_[1]))
            used |= {a_, r_[1]}
            break
    # pair 3: two demanded skills that no occupation needs together
    z = next(((a_, b_) for a_ in top[:30] for b_ in top[:30] if a_ != b_ and not {a_, b_} & used
              and cond(a_, b_)[0] == 0), None)
    if z:
        pairs.append(("p3", z[0], z[1]))

    dx = fig.add_axes([0.05, 0.03, 0.9, 0.3])
    dx.set_xlim(0, 100)
    dx.set_ylim(0, 40)
    dx.axis("off")
    dx.text(-1, 39.5, "d", fontsize=13, fontweight="bold", color=INK, va="top")

    def node(x, y, label, col, r=3.2, tag=""):
        dx.scatter([x], [y], s=(r * 70) ** 1.0 * 2.2, facecolors="white", edgecolors=col, linewidths=1.4, zorder=4)
        dx.text(x, y, tag, ha="center", va="center", fontsize=8, fontweight="bold", color=col, zorder=5)
        dx.text(x, y - r - 0.6, short(label, 24), ha="center", va="top", fontsize=6.0, color=col, zorder=5)

    def nest_arrow(x0, x1, y, txt, col_from, col_to, width=4.5):
        dx.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle=f"simple,head_width={width},head_length=4,tail_width={width * 0.55}",
                                     mutation_scale=2.2, fc="#d4d3cc", ec="none", zorder=2))
        dx.text((x0 + x1) / 2, y, txt, ha="center", va="center", fontsize=6.3, color=INK, zorder=3)

    for kind, a_, b_ in pairs:
        n_ab, n_b = cond(a_, b_)
        n_ba, n_a = cond(b_, a_)
        rel_ = "<" if n_ab / max(n_b, 1) < n_ba / max(n_a, 1) else ">"
        la, lb = slabel.get(a_, ""), slabel.get(b_, "")
        ca = {"specific": SPEC, "intermediate": INTER, "general": GEN}.get(gmap.get(a_), SPEC)
        cb = {"specific": SPEC, "intermediate": INTER, "general": GEN}.get(gmap.get(b_), GEN)
        if kind == "big":
            dx.add_patch(FancyBboxPatch((1, 1), 58, 36, boxstyle="round,pad=0.3", fc="none", ec=MUTED, lw=0.7, ls="--"))
            oa = [o_ for o_ in has.get(a_, set()) if o_ in olab][:3] + [o_ for o_ in has.get(a_, set()) if o_ not in olab][:2]
            ob = [o_ for o_ in has.get(b_, set()) if o_ not in has.get(a_, set())][:3]
            shown = oa + ob
            names_ = {o_: olab.get(o_, occs.set_index("conceptUri").preferredLabel.get(o_, "")) for o_ in shown}
            xs = np.linspace(8, 52, max(len(shown), 1))
            node(9, 9, la, ca, tag="A")
            node(51, 9, lb, cb, tag="B")
            for x_, o_ in zip(xs, shown):
                inA = o_ in has.get(a_, set())
                fc = "#e9ecf9" if inA else "#f2f1ee"
                dx.add_patch(FancyBboxPatch((x_ - 4.2, 27), 8.4, 7, boxstyle="round,pad=0.2", fc=fc, ec=MUTED, lw=0.5))
                nm_ = short(names_[o_], 26)
                if len(nm_) > 13 and " " in nm_:
                    cut = nm_.rfind(" ", 0, 14)
                    nm_ = nm_[:cut] + "\n" + nm_[cut + 1:] if cut > 0 else nm_
                dx.text(x_, 30.5, nm_, ha="center", va="center", fontsize=4.9, color=INK, linespacing=1.0)
                if inA:
                    added = EDIT.get((o_, a_)) == "added"
                    dx.plot([9, x_], [12.2, 26.8], color=GREEN if added else SPEC, lw=1.0 if added else 0.6,
                            ls="--" if added else "-", zorder=1)
                if o_ in has.get(b_, set()):
                    dx.plot([51, x_], [12.2, 26.8], color=MUTED, lw=0.5, zorder=1)
            nest_arrow(46, 14, 9, "", cb, ca)
            dx.text(30, 15.5, f"P(A | B) = {n_ab}/{n_b}  {rel_}  P(B | A) = {n_ba}/{n_a}", ha="center", va="center",
                    fontsize=7, color=INK, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
            dx.text(30, 12.6, (f"A is nested in B: {n_ba} of {n_a} occupations with A also have B; "
                               f"only {n_ab} of {n_b} with B have A") if rel_ == "<" else "B is nested in A", ha="center", fontsize=5.6, color=INK2, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            dx.text(30, 2.2, "green dashed = link added by ZenESCO; probabilities over all ESCO occupations, "
                    "updated graph", ha="center", fontsize=5.4, color=GREEN)
        else:
            y0 = 28 if kind == "p2" else 9
            dx.add_patch(FancyBboxPatch((62, y0 - 8), 37, 17, boxstyle="round,pad=0.3", fc="none", ec=MUTED, lw=0.7, ls="--"))
            t1, t2 = ("C", "D") if kind == "p2" else ("E", "F")
            node(69, y0 + 1, short(la, 18), ca, r=2.6, tag=t1)
            node(92, y0 + 1, short(lb, 18), cb, r=2.6, tag=t2)
            txt = (f"P({t1} | {t2}) = {n_ab}/{n_b}  {rel_}  P({t2} | {t1}) = {n_ba}/{n_a}"
                   if n_ab else f"P({t1} | {t2}) = 0: never needed together")
            if n_ab:
                nest_arrow(89.5, 71.5, y0 + 1, "", cb, ca, width=3.5)
            else:
                dx.plot([71.5, 89.5], [y0 + 1, y0 + 1], color=GRID, lw=6, solid_capstyle="round", zorder=2)
            dx.text(80.5, y0 + 6.6, txt, ha="center", va="center", fontsize=6.4, color=INK, fontweight="bold")
    save(fig, "F9b_skill_groups_nestedness")


# =====================================================================================
# F9c  demand over time
# =====================================================================================
def f9c():
    a = ads[["ad_id", "occ_uri", "upload_date", "window"]].copy()
    a["month"] = pd.to_datetime(a.upload_date, errors="coerce").dt.to_period("M")
    a = a[a.month.notna()]
    m = ask.merge(a[["ad_id", "month"]], on="ad_id")

    def cat(o, s_):
        u, og = UPD[o], ORIG[o]
        if s_ in u and s_ not in og:
            return "added by ZenESCO"
        if s_ in og and s_ not in u:
            return "deleted by ZenESCO"
        if s_ in u:
            return "ESCO essential" if u[s_] == 1.0 else "ESCO optional"
        if s_ in coll:
            return "language & transversal (outside profiles)"
        return "not in the profile"

    m["cat"] = [cat(o, s_) for o, s_ in zip(m.occ_uri, m.skill_uri)]
    m["edit"] = [EDIT.get((o, s_), "") for o, s_ in zip(m.occ_uri, m.skill_uri)]
    n_ads = a.groupby("month").size()
    order = ["ESCO essential", "ESCO optional", "added by ZenESCO", "deleted by ZenESCO",
             "language & transversal (outside profiles)", "not in the profile"]
    cols = {"ESCO essential": BLUE, "ESCO optional": "#9cc3f0", "added by ZenESCO": GREEN,
            "deleted by ZenESCO": RED, "language & transversal (outside profiles)": YELLOW, "not in the profile": "#d4d3cc"}
    tab = m.groupby(["month", "cat"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    tab = tab.div(n_ads.reindex(tab.index), axis=0)
    months = [str(p) for p in tab.index]
    x = np.arange(len(months))
    held = np.array([str(p) >= "2025-01" for p in tab.index])
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.4), gridspec_kw=dict(wspace=0.38))
    ax = axs[0]
    ax.stackplot(x, [tab[c].values for c in order], colors=[cols[c] for c in order], alpha=0.9, lw=0)
    if held.any():
        x0 = x[held][0] - 0.5
        ytop = tab.sum(axis=1).max()
        ax.add_patch(Rectangle((x0, 0), x[-1] - x0 + 0.5, ytop * 1.22, fc="none",
                               hatch="///", ec="#8a8984", lw=0, alpha=0.35, zorder=3))
        ax.set_ylim(0, ytop * 1.22)
        ax.text(x0 + 0.25, ytop * 1.19, "held-out months\n(not used to choose edits)", fontsize=6,
                color=INK, va="top", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.set_xticks(x)
    nlab = [f"{mo[2:].replace('-', '/')}  n={int(n_ads.get(p_, 0))}" for mo, p_ in zip(months, tab.index)]
    ax.set_xticklabels(nlab, rotation=90, fontsize=5.9)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("Skill mentions per ad, by how the\nupdated ESCO profile treats them")
    ax.set_title("Skill demand in the analysed\noccupations' ads", fontsize=8, fontweight="bold")
    h1 = tab[held].sum() / tab[held].sum().sum() if held.any() else tab.sum() / tab.sum().sum()
    labels = [f"{c} ({h1[c]:.0%})" for c in order]
    ax.legend([Rectangle((0, 0), 1, 1, fc=cols[c]) for c in order][::-1], labels[::-1], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), fontsize=5.8, ncol=2, title="share of mentions in held-out months",
              title_fontsize=5.8)
    letter(ax, "a", -0.22, 1.02)
    # (b) mentions of edited skills per 100 ads
    bx = axs[1]
    ecols = {"added": GREEN, "upgraded": VIOLET, "demoted": ORANGE, "deleted": RED}
    et = m[m.edit != ""].groupby(["month", "edit"]).size().unstack(fill_value=0).reindex(
        index=tab.index, columns=list(ecols), fill_value=0)
    et = et.div(n_ads.reindex(et.index), axis=0) * 100
    bottom = np.zeros(len(et))
    for k_, c_ in ecols.items():
        bx.bar(x, et[k_].values, bottom=bottom, color=c_, width=0.75, label=k_, edgecolor="white", lw=0.4,
               hatch=None)
        bottom += et[k_].values
    if held.any():
        for xi in x[held]:
            bx.add_patch(Rectangle((xi - 0.375, 0), 0.75, bottom[xi], fc="none", ec="white", hatch="////", lw=0, alpha=0.6))
    bx.set_xticks(x)
    bx.set_xticklabels(nlab, rotation=90, fontsize=5.9)
    bx.set_ylim(0, bottom.max() * 1.08)
    bx.set_ylabel("Mentions per 100 ads of the\nskills ZenESCO edits")
    bx.set_title("How often the edited skills\nappear in ads, by edit type", fontsize=8, fontweight="bold")
    tot = et[held].mean() if held.any() else et.mean()
    bx.legend([Rectangle((0, 0), 1, 1, fc=c) for c in ecols.values()][::-1],
              [f"{k} ({tot[k]:.1f} per 100 ads)" for k in ecols][::-1], loc="upper center", bbox_to_anchor=(0.5, -0.2),
              fontsize=5.8, ncol=2, title="held-out average", title_fontsize=5.8)
    bx.grid(axis="y", color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    letter(bx, "b", -0.22, 1.02)
    save(fig, "F9c_demand_over_time")


for f in (f9a, f9b, f9c):
    try:
        f()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"{f.__name__} failed: {ex!r}")
print(f"Figures in {OUT}")
