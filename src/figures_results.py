"""
ZenESCO - result figures after Stage 7 (replace the network figure F4c)
======================================================================

F4c  Decision map (style of the nestedness-scatter reference figure). Four panels, each answers one question:
     a  Where does every ESCO link sit on the two evidence axes, and what does ZenESCO do with it?
        x = share of the occupation's ads mentioning the skill, relative to what an UNRELATED skill reaches
            (1 = unrelated-skill level N_o); y = probability that the skill is over-represented in this
            occupation's ads (specificity). The decision thresholds are drawn: y = 0.95 (supported),
            y = 0.5 and x = 1 (weak). Never-mentioned links sit in a strip on the left.
     b  The same map for skills NOT in ESCO: which pass the addition screen (specific AND as common as the
        occupation's optional links, x = share relative to R_opt).
     c  Does each candidate edit survive the held-out test (Stage 7)? Change in identification (MRR) on
        H1 2025 ads, with the non-inferiority margin m_U.
     d  Before vs after: how many of the occupation's 20 most demanded skills in HELD-OUT ads (H1 2025)
        are in its ESCO profile.
F10  Old and new skills on a skill map (style of the "old vs new skills" reference figure), for two
     occupations chosen by rule (most ads in different clusters), plus held-out coverage by skill rank.
     Map = t-SNE of which ESCO occupations use each skill (TF-IDF of the skill x occupation matrix, SVD);
     regions coloured and labelled by the ISCO occupation group that uses the skill most.

Edits drawn as "applied" = Stage 6 candidates that pass Stage 7 (deletions/demotions non-inferior; additions
and upgrades with a 90% interval above 0). Final recommendations come in Stage 9.

Run   python figures_results.py                                   # BASE = C:\\skillab_data
      python figures_results.py C:\\skillab_data "quality engineer" "purchaser"   # other F10 occupations
Needs pandas, numpy, matplotlib, scikit-learn (pip install scikit-learn)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
F10_OCC = sys.argv[2:4] if len(sys.argv) > 3 else None
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ESCO = BASE / "esco"

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#008300", "#6250d6", "#e34948", "#e87ba4")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
ACT_STYLE = {  # action -> (colour, marker, label)
    "retain: supported/adequate": (BLUE, "o", "retain: supported / adequate"),
    "retain: protected": ("#7d7c76", "s", "retain: protected"),
    "retain: implied": ("#a9a8a1", "D", "retain: implied by a related skill"),
    "retain: no evidence": ("#cfcec7", "o", "retain: absence not informative"),
    "DELETE candidate": (RED, "X", "delete candidate"),
    "DEMOTE candidate": (ORANGE, "v", "demote candidate (essential → optional)"),
    "UPGRADE candidate": (VIOLET, "^", "upgrade candidate (optional → essential)")}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=SURF)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {name}")


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.3, color=INK, fontweight="bold", pad=6)


def short(t, n=30):
    t = str(t)
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def esco_file(name):
    hits = [p for p in ESCO.glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    return hits[0] if hits else None


# ---- data ---------------------------------------------------------------------------
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
ADD = pd.read_parquet(BASE / "stage6" / "add_candidates.parquet")
PE = pd.read_parquet(BASE / "stage4" / "pair_evidence.parquet")
EF = pd.read_parquet(BASE / "stage7" / "edit_effects.parquet")
m_U = float(pd.read_csv(BASE / "stage7" / "baseline.csv").m_U.iloc[0])
REFS = EV.groupby("occ_uri")[["N_null", "R_opt", "R_ess", "R_dem"]].first()
olab = EV.groupby("occ_uri").occupation.first()
ocl = EV.groupby("occupation").cluster.first()

passed = {}
for o, s_, a_, imp, ni in zip(EF.occupation, EF.skill_uri, EF.action, EF.improves, EF.noninferior):
    passed[(o, s_, a_)] = bool(imp) if a_ in ("ADD candidate", "UPGRADE candidate") else bool(ni)

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


def is_applied(o, s_, a_):
    return passed.get((o, s_, a_), False)


# held-out (H1 2025) demand per occupation, for panel d and F10
ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet", columns=["ad_id", "window", "analysable"])
ads = ads[ads.analysable & (ads.window == "H1_2025")]
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
ads = ads.merge(occ_of, on="ad_id")
ads = ads[ads.occ_uri.isin(set(EV.occ_uri))]
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(ads.ad_id))].drop_duplicates().merge(ads[["ad_id", "occ_uri"]], on="ad_id")
N1 = ads.groupby("occ_uri").size()
H1 = (ask.groupby(["occ_uri", "skill_uri"]).ad_id.nunique() / ask.occ_uri.map(N1).groupby(
    [ask.occ_uri, ask.skill_uri]).first()).rename("prev_h1").reset_index()
skills = pd.read_csv(esco_file("skills_en.csv"), dtype=str, keep_default_na=False)
slabel = dict(zip(skills.conceptUri, skills.preferredLabel))
lang = set(pd.read_csv(esco_file("languageSkillsCollection_en.csv"), dtype=str).conceptUri) \
    if esco_file("languageSkillsCollection_en.csv") else set()


def profiles(o):
    """original and updated profile (skill -> weight) of one occupation"""
    e = EV[EV.occ_uri == o]
    orig = {s_: (1.0 if r == "essential" else 0.5) for s_, r in zip(e.skill_uri, e.relation)}
    upd = dict(orig)
    on = olab[o]
    for s_, a_ in zip(e.skill_uri, e.action):
        if not is_applied(on, s_, a_):
            continue
        if a_ == "DELETE candidate":
            upd.pop(s_, None)
        elif a_ == "DEMOTE candidate":
            upd[s_] = 0.5
        elif a_ == "UPGRADE candidate":
            upd[s_] = 1.0
    for s_, t in zip(ADD[ADD.occ_uri == o].skill_uri, ADD[ADD.occ_uri == o].proposed_type):
        if is_applied(on, s_, "ADD candidate"):
            upd[s_] = 1.0 if t == "essential" else 0.5
    return orig, upd


# =====================================================================================
# F4c decision map
# =====================================================================================
def f4c():
    fig = plt.figure(figsize=(7.4, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.34, width_ratios=[1.15, 1])
    # ---- (a) every ESCO link
    ax = fig.add_subplot(gs[0, 0])
    e = EV.copy()
    e["xr"] = e.prev_raw / e.N_null.clip(lower=1e-4)
    rng = np.random.default_rng(1)
    STRIP = 0.03
    e["xp"] = np.where(e.n > 0, e.xr.clip(lower=0.06), STRIP * np.exp(rng.uniform(-0.35, 0.35, len(e))))
    e["yp"] = e.pr_lift.clip(0, 1) + np.where(e.n > 0, 0, rng.uniform(-0.015, 0.015, len(e)))
    ax.add_patch(Rectangle((0.045, 0.95), 1e4, 0.07, fc=BLUE, alpha=0.07, lw=0, zorder=0))
    ax.add_patch(Rectangle((0.045, -0.03), 1 - 0.045, 0.53, fc=RED, alpha=0.06, lw=0, zorder=0))
    ax.add_patch(Rectangle((0.018, -0.03), 0.027, 1.06, fc="#f2f1ee", lw=0, zorder=0))
    order = ["retain: no evidence", "retain: implied", "retain: protected", "retain: supported/adequate",
             "DEMOTE candidate", "DELETE candidate", "UPGRADE candidate"]
    for a_ in order:
        g = e[e.action == a_]
        col, mk, lab = ACT_STYLE[a_]
        ax.scatter(g.xp, g.yp, s=13 if mk in "Xv^" else 9, marker=mk, c=col, alpha=0.75, lw=0,
                   label=f"{lab} ({len(g)})", zorder=3 if a_.startswith("retain") else 4)
    ax.set_xscale("log")
    ax.set_xlim(0.018, max(e.xp.max() * 1.5, 30))
    ax.set_ylim(-0.03, 1.02)
    ax.axhline(0.95, color=BLUE, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0.5, color=RED, lw=0.6, ls=(0, (1, 1.5)))
    ax.axvline(1, color=RED, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(STRIP, 0.12, f"never\nmentioned\n({int((e.n == 0).sum())})", ha="center", va="bottom", fontsize=5.4,
            color=INK2, linespacing=1.0, zorder=6, bbox=dict(boxstyle="round,pad=0.1", fc="#f2f1ee", ec="none"))
    ax.text(ax.get_xlim()[1] * 0.85, 0.915, "supported\n(Pr ≥ 0.95)", ha="right", va="top",
            fontsize=5.8, color=BLUE, linespacing=1.0)
    ax.text(0.06, 0.03, "weak: no more common than an\nunrelated skill and not specific", ha="left",
            va="bottom", fontsize=5.8, color=RED, linespacing=1.05)
    ax.set_xlabel("share of the occupation's ads mentioning the skill\n(× the level an unrelated skill reaches)")
    ax.set_ylabel("specificity: Pr(skill over-represented\nin this occupation's ads)")
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    panel(ax, "a", "Every ESCO link on the two evidence axes")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.3), fontsize=5.7, ncol=1, markerscale=1.3,
              handletextpad=0.2, labelspacing=0.25)
    # ---- (b) skills not in ESCO
    bx = fig.add_subplot(gs[0, 1])
    p = PE[(PE.relation == "") & (PE.n >= 2)].copy()
    p = p.join(REFS, on="occ_uri")
    p["xr"] = p.prev_raw / p.R_opt.clip(lower=1e-4)
    key_add = set(zip(ADD.occ_uri, ADD.skill_uri))
    p["is_add"] = [(o, s_) in key_add for o, s_ in zip(p.occ_uri, p.skill_uri)]
    p["lang"] = p.skill_uri.isin(lang)
    xmax = max(p.xr.max() * 1.6, 5)
    bx.add_patch(Rectangle((1, 0.95), xmax, 0.07, fc=GREEN, alpha=0.08, lw=0, zorder=0))
    q = p[~p.is_add & ~p.lang]
    bx.scatter(q.xr, q.pr_lift.clip(0, 1), s=3, c="#b9b8b0", alpha=0.35, lw=0, rasterized=True,
               label=f"rejected ({len(q):,})")
    q = p[p.lang]
    bx.scatter(q.xr, q.pr_lift.clip(0, 1), s=7, marker="s", c=YELLOW, alpha=0.8, lw=0,
               label=f"language skill: outside profiles by design ({len(q)})")
    q = p[p.is_add]
    applied_add = [is_applied(olab[o], s_, "ADD candidate") for o, s_ in zip(q.occ_uri, q.skill_uri)]
    bx.scatter(q.xr, q.pr_lift.clip(0, 1), s=11, c=np.where(applied_add, GREEN, "#9fd39f"), lw=0,
               label=f"addition candidate ({len(q)}); dark = passes Stage 7")
    top = q.assign(ap=applied_add).sort_values("prev_raw", ascending=False).drop_duplicates("skill_uri").head(7)
    for i, r in enumerate(top.itertuples()):
        bx.annotate(f"{short(slabel.get(r.skill_uri, ''), 24)} · {short(olab[r.occ_uri], 14)}",
                    xy=(r.xr, min(r.pr_lift, 1)), xytext=(0.98, 0.82 - 0.075 * i), textcoords="axes fraction",
                    ha="right", va="center", fontsize=5.3, color=GREEN,
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.35, alpha=0.6))
    bx.set_xscale("log")
    bx.set_xlim(max(p.xr[p.xr > 0].min() * 0.7, 1e-3), xmax)
    bx.axhline(0.95, color=GREEN, lw=0.7, ls=(0, (3, 2)))
    bx.axvline(1, color=GREEN, lw=0.7, ls=(0, (3, 2)))
    bx.set_ylim(-0.03, 1.02)
    bx.set_xlabel("share of the occupation's ads mentioning the skill\n(× the level of its confirmed optional links)")
    bx.set_ylabel("specificity")
    bx.text(0.98, 0.03, "added only if specific (≥ 0.95)\nAND as common as the confirmed\noptional links (≥ 1)",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=5.8, color=GREEN, linespacing=1.05)
    panel(bx, "b", "Skills the ads ask for that ESCO lacks")
    bx.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.3), fontsize=5.7, markerscale=1.6, handletextpad=0.2,
              labelspacing=0.25)
    # ---- (c) held-out test of each candidate
    cx = fig.add_subplot(gs[1, 0])
    acts = ["DELETE candidate", "DEMOTE candidate", "UPGRADE candidate", "ADD candidate"]
    names = ["delete", "demote", "upgrade", "add"]
    cols = [RED, ORANGE, VIOLET, GREEN]
    for i, (a_, c_) in enumerate(zip(acts, cols)):
        g = EF[EF.action == a_]
        if g.empty:
            continue
        ok = g.improves if a_ in ("ADD candidate", "UPGRADE candidate") else g.noninferior
        x = i + rng.uniform(-0.28, 0.28, len(g))
        cx.scatter(x, g.dU * 1000, s=8, c=np.where(ok, c_, "#cfcec7"), lw=0, alpha=0.85, zorder=3)
        names[i] = f"{names[i]}\n{ok.mean():.0%} pass\nn = {len(g)}"
    cx.axhline(0, color=INK2, lw=0.6)
    cx.axhline(m_U * 1000, color=RED, lw=0.7, ls=(0, (3, 2)))
    cx.text(-0.55, m_U * 1000, " non-inferiority margin m_U", fontsize=5.6, color=RED, va="top", ha="left")
    cx.set_xticks(range(4))
    cx.set_xticklabels(names, fontsize=6.4, linespacing=1.1)
    cx.set_ylabel("change in identification on held-out ads\n(ΔMRR × 1000, H1 2025)")
    cx.set_xlim(-0.6, 3.6)
    panel(cx, "c", "Does each edit survive the held-out test?")
    cx.text(0, -0.36, "grey = fails: deletion/demotion below the margin, or addition/upgrade without a clear gain",
            transform=cx.transAxes, fontsize=5.6, color=MUTED)
    # ---- (d) top-20 held-out demand covered, before vs after
    dx = fig.add_subplot(gs[1, 1])
    rows = []
    for o in EV.occ_uri.unique():
        h = H1[(H1.occ_uri == o) & ~H1.skill_uri.isin(lang)].nlargest(20, "prev_h1")
        orig, upd = profiles(o)
        rows.append((olab[o], ocl[olab[o]], h.skill_uri.isin(set(orig)).mean() * 100,
                     h.skill_uri.isin(set(upd)).mean() * 100))
    d = pd.DataFrame(rows, columns=["occ", "cl", "before", "after"]).sort_values(["cl", "before"])
    y = np.arange(len(d))
    for yi, b, a in zip(y, d.before, d.after):
        dx.plot([b, a], [yi, yi], color=GREEN if a > b else GRID, lw=1.6, zorder=2)
    dx.scatter(d.before, y, s=22, c="white", edgecolors=INK2, lw=0.9, zorder=3, label="original ESCO")
    dx.scatter(d.after, y, s=22, c=GREEN, zorder=4, label="after ZenESCO edits")
    dx.set_yticks(y)
    dx.set_yticklabels([short(o, 24) for o in d.occ], fontsize=6.1)
    dx.set_xlabel("% of the 20 most demanded skills in held-out ads\nthat the ESCO profile contains")
    dx.set_xlim(-2, 102)
    dx.grid(axis="x", color=GRID, lw=0.5)
    dx.legend(loc="lower right", fontsize=6)
    panel(dx, "d", "Coverage of held-out demand")
    save(fig, "F4c_decision_map")


# =====================================================================================
# F10 old and new skills on a skill map
# =====================================================================================
def skill_groups(uris):
    h = pd.read_csv(esco_file("skillsHierarchy_en.csv"), dtype=str)
    lvl1 = dict(zip(h["Level 1 URI"].dropna(), h["Level 1 preferred term"].dropna()))
    lvl0 = dict(zip(h["Level 0 URI"].dropna(), h["Level 0 preferred term"].dropna()))
    b = pd.read_csv(esco_file("broaderRelationsSkillPillar_en.csv"), dtype=str)
    par = b.groupby("conceptUri").broaderUri.agg(list).to_dict()
    out = {}
    for s_ in uris:
        seen, front, hit = set(), [s_], None
        for _ in range(10):
            nxt = []
            for x in front:
                for pp in par.get(x, []):
                    if pp in lvl1:
                        hit = lvl1[pp]
                        break
                    if pp not in seen:
                        seen.add(pp)
                        nxt.append(pp)
                if hit:
                    break
            if hit or not nxt:
                break
            front = nxt
        out[s_] = hit or "other"
    short_names = {"working with machinery and specialised equipment": "machinery & equipment",
                   "communication, collaboration and creativity": "communication & collaboration",
                   "engineering, manufacturing and construction": "engineering & manufacturing (knowledge)",
                   "information and communication technologies (icts)": "ICT (knowledge)",
                   "business, administration and law": "business, admin & law (knowledge)",
                   "natural sciences, mathematics and statistics": "sciences & maths (knowledge)",
                   "working with computers": "working with computers", "management skills": "management",
                   "information skills": "information skills", "assisting and caring": "assisting & caring",
                   "handling and moving": "handling & moving", "constructing": "constructing"}
    return {k: short_names.get(v, "other") for k, v in out.items()}


def f10():
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
    from scipy import sparse
    # skills are placed by WHICH ESCO OCCUPATIONS USE THEM: skills of engineers, managers, trades, ICT ... form regions
    rel = pd.read_csv(esco_file("occupationSkillRelations_en.csv"), dtype=str, usecols=["occupationUri", "skillUri"])
    occ = pd.read_csv(esco_file("occupations_en.csv"), dtype=str, usecols=["conceptUri", "iscoGroup"])
    pool = set(EV.skill_uri) | set(ADD.skill_uri) | set(PE[PE.n >= 3].skill_uri)
    pool = sorted(s_ for s_ in pool if s_ in set(rel.skillUri))
    si_ = {s_: i for i, s_ in enumerate(pool)}
    oc_ = {o_: i for i, o_ in enumerate(sorted(rel.occupationUri.unique()))}
    r = rel[rel.skillUri.isin(si_)]
    U = sparse.csr_matrix((np.ones(len(r)), (r.skillUri.map(si_).values, r.occupationUri.map(oc_).values)),
                          shape=(len(pool), len(oc_)))
    Z = normalize(TruncatedSVD(n_components=50, random_state=0).fit_transform(TfidfTransformer().fit_transform(U)))
    try:
        xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=0).fit_transform(Z)
    except TypeError:
        xy = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(Z)
    M = pd.DataFrame(xy, index=pool, columns=["x", "y"])
    region_of = {"21": "Engineering & science", "31": "Engineering technicians",
                 "11": "Management", "12": "Management", "13": "Management", "14": "Management",
                 "25": "ICT", "35": "ICT", "24": "Business & administration", "33": "Business & administration",
                 "41": "Business & administration", "42": "Business & administration", "43": "Logistics & stock",
                 "44": "Business & administration", "71": "Building trades", "72": "Metal & machinery trades",
                 "73": "Metal & machinery trades", "74": "Electrical & electronics trades", "75": "Food, wood & textile trades",
                 "81": "Plant & machine operators", "82": "Plant & machine operators", "83": "Logistics & stock",
                 "22": "Health, education & services", "32": "Health, education & services",
                 "23": "Health, education & services", "26": "Health, education & services",
                 "34": "Health, education & services", "51": "Health, education & services",
                 "52": "Health, education & services", "53": "Health, education & services",
                 "54": "Health, education & services"}
    isco2 = occ.set_index("conceptUri").iscoGroup.str[:2]
    r2 = r.assign(reg=r.occupationUri.map(isco2).map(region_of).fillna("Other"))
    cnt_ = r2.groupby(["skillUri", "reg"]).size().reset_index(name="k").sort_values("k", ascending=False)
    M["grp"] = cnt_.drop_duplicates("skillUri").set_index("skillUri").reg.reindex(M.index).fillna("Other")
    # one label per region, at the densest spot of that region; skipped if it would sit on another label
    span = np.hypot(np.ptp(M.x), np.ptp(M.y))
    region_labels = []
    for g_, gg in sorted(M.groupby("grp"), key=lambda t: -len(t[1])):
        if len(gg) < 25 or g_ == "Other":
            continue
        H, xe, ye = np.histogram2d(gg.x, gg.y, bins=18, range=[[M.x.min(), M.x.max()], [M.y.min(), M.y.max()]])
        ix, iy = np.unravel_index(H.argmax(), H.shape)
        if H[ix, iy] < 6:
            continue
        lx, ly = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
        if any(np.hypot(lx - a, ly - b) < 0.12 * span for a, b, _ in region_labels):
            continue
        region_labels.append((lx, ly, g_))
    gcol = {"Engineering & science": "#cfe0f7", "Engineering technicians": "#d9e8fb", "Management": "#f7dccc",
            "ICT": "#c8ebe0", "Business & administration": "#f8e7c4", "Logistics & stock": "#e9e2f7",
            "Metal & machinery trades": "#dcebd2", "Electrical & electronics trades": "#d3eef0",
            "Building trades": "#ece0d0", "Plant & machine operators": "#e4e4dc",
            "Food, wood & textile trades": "#efe6d8", "Health, education & services": "#f5d6e3", "Other": "#eeeeec"}
    if F10_OCC:
        occs = F10_OCC
    else:  # rule fixed before results: the occupation with most ads in each of the two largest clusters
        n_ads = EV.groupby("occupation").N.first()
        top = n_ads.groupby(EV.groupby("occupation").cluster.first()).idxmax()
        occs = list(n_ads[top.values].sort_values(ascending=False).index[:2])
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.75], hspace=0.5, wspace=0.18)
    for j, on in enumerate(occs):
        o = EV[EV.occupation == on].occ_uri.iloc[0]
        orig, upd = profiles(o)
        e = EV[EV.occ_uri == o].set_index("skill_uri")
        ax = fig.add_subplot(gs[0, j])
        for g_, gg in M.groupby("grp"):
            ax.scatter(gg.x, gg.y, s=9, c=gcol.get(g_, "#eeeeec"), lw=0, zorder=1, rasterized=True)
        for lx, ly, g_ in region_labels:
            ax.text(lx, ly, g_.replace(" & ", " &\n", 1) if len(g_) > 18 else g_, fontsize=5.8, color=INK2,
                    ha="center", va="center", style="italic", fontweight="bold", zorder=7, linespacing=1.0,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        kept = [s_ for s_ in orig if s_ in upd and s_ in M.index]
        dele = [s_ for s_ in orig if s_ not in upd and s_ in M.index]
        added = [s_ for s_ in upd if s_ not in orig and s_ in M.index]
        upg = [s_ for s_ in kept if upd[s_] > orig[s_]]
        dem = [s_ for s_ in kept if upd[s_] < orig[s_]]
        ax.scatter(M.loc[kept, "x"], M.loc[kept, "y"], s=[26 if orig[s_] == 1 else 13 for s_ in kept],
                   c=BLUE, edgecolors="white", lw=0.4, zorder=4, label=f"ESCO skills kept ({len(kept)})")
        ax.scatter(M.loc[dele, "x"], M.loc[dele, "y"], s=22, marker="X", c=RED, lw=0, zorder=5,
                   label=f"deleted ({len(dele)})")
        ax.scatter(M.loc[dem, "x"], M.loc[dem, "y"], s=24, marker="v", c=ORANGE, lw=0, zorder=5,
                   label=f"demoted ({len(dem)})")
        ax.scatter(M.loc[upg, "x"], M.loc[upg, "y"], s=40, marker="o", facecolors="none", edgecolors=VIOLET,
                   lw=1.3, zorder=6, label=f"upgraded ({len(upg)})")
        ax.scatter(M.loc[added, "x"], M.loc[added, "y"], s=34, c=GREEN, edgecolors="white", lw=0.5, zorder=6,
                   label=f"added ({len(added)})")

        def centroid(prof):
            ks = [s_ for s_ in prof if s_ in M.index]
            w = np.array([prof[s_] for s_ in ks])
            return (M.loc[ks, ["x", "y"]].values * w[:, None]).sum(0) / w.sum()
        c0, c1 = centroid(orig), centroid(upd)
        ax.annotate("", xy=c1, xytext=c0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=10),
                    zorder=8)
        ax.scatter(*c0, s=60, marker="*", c=INK, zorder=8)
        lab_s = (sorted(added, key=lambda s_: -ADD[(ADD.occ_uri == o) & (ADD.skill_uri == s_)].prev_raw.max())[:4]
                 + sorted(dele, key=lambda s_: -e.loc[s_, "generality"])[:3])
        # labels in two columns at the map edges, ordered by height, so they never overlap
        xmid = M.x.median()
        lab_s = list(dict.fromkeys(lab_s))
        for side in ("L", "R"):
            grp_ = [s_ for s_ in lab_s if (M.loc[s_, "x"] < xmid) == (side == "L")]
            grp_ = sorted(grp_, key=lambda s_: -M.loc[s_, "y"])
            y0_, y1_ = ax.get_ylim()
            ys_ = [min(max((M.loc[s_, "y"] - y0_) / (y1_ - y0_), 0.05), 0.95) for s_ in grp_]
            for k in range(1, len(ys_)):                     # push apart, top to bottom
                ys_[k] = min(ys_[k], ys_[k - 1] - 0.05)
            if ys_ and ys_[-1] < 0.04:                         # shift up if the stack ran off the bottom
                sh_ = 0.04 - ys_[-1]
                ys_ = [v + sh_ for v in ys_]
            for s_, yt in zip(grp_, ys_):
                col = GREEN if s_ in added else RED
                ax.annotate(short(slabel[s_], 24), xy=(M.loc[s_, "x"], M.loc[s_, "y"]),
                            xytext=(0.01 if side == "L" else 0.99, yt + 0.025), textcoords="axes fraction", fontsize=5.4,
                            color=col, ha="left" if side == "L" else "right", va="center", zorder=9,
                            arrowprops=dict(arrowstyle="-", color=col, lw=0.4, alpha=0.7, shrinkB=2),
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
        panel(ax, "ab"[j], on)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=5.6, markerscale=0.9,
                  handletextpad=0.2, columnspacing=0.8)

        # coverage by rank of held-out demand
        bx = fig.add_subplot(gs[1, j])
        h = H1[(H1.occ_uri == o) & ~H1.skill_uri.isin(lang)].sort_values("prev_h1", ascending=False).head(60)
        r = np.arange(1, len(h) + 1)
        cb = np.cumsum(h.skill_uri.isin(set(orig)).values) / r * 100
        ca = np.cumsum(h.skill_uri.isin(set(upd)).values) / r * 100
        bx.fill_between(r, 0, ca, color=GREEN, alpha=0.25, lw=0, step="mid")
        bx.fill_between(r, 0, cb, color=BLUE, alpha=0.35, lw=0, step="mid")
        bx.plot(r, ca, color=GREEN, lw=1.3, drawstyle="steps-mid", label="after ZenESCO edits")
        bx.plot(r, cb, color=BLUE, lw=1.3, drawstyle="steps-mid", label="original ESCO")
        for k in (10, 30):
            if k <= len(r):
                bx.annotate(f"top {k}: {cb[k - 1]:.0f}% → {ca[k - 1]:.0f}%", xy=(k, ca[k - 1]), xytext=(4, 6),
                            textcoords="offset points", fontsize=5.8, color=INK)
                bx.axvline(k, color=GRID, lw=0.6)
        bx.set_ylim(0, 100)
        bx.set_xlim(1, len(r))
        bx.set_xlabel("skills ranked by demand in held-out ads (H1 2025)")
        if j == 0:
            bx.set_ylabel("% of the top-k skills\nin the ESCO profile")
        bx.legend(loc="upper right", fontsize=5.8)
        panel(bx, "cd"[j], "Held-out demand covered")
    fig.text(0.5, 0.005, "Map: t-SNE of which ESCO occupations use each skill; background = skills of the analysed occupations and their\n"
             "ads, coloured by the occupation group that uses them most. ★ = centre of the original profile; arrow = shift after the edits.",
             ha="center", fontsize=5.8, color=MUTED)
    save(fig, "F10_old_and_new_skills" + ("" if not F10_OCC else "_" + "_".join(x.split()[0] for x in occs)))


for f in (f4c, f10):
    try:
        f()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"{f.__name__} failed: {ex!r}")
print(f"Figures in {OUT}")
