"""
ZenESCO - paper figures that the data of Stages 1-6 can support (F1, F2, F3, F4, F6)
====================================================================================

    F1  Architecture: official, empirical and updated layers over the same nodes (style of the
        multilayer-network reference figure N9). Schematic, no data.
    F2  What ESCO looks like for the analysed occupations: (a) profile size by occupation and cluster,
        essential vs optional; (b) skill generality; (c) what kind of skills the links point to.
    F3  Digital gap (reference figure D): digital share of the ESCO profile vs of the skills that are
        specific to the occupation's ads, by cluster; arrows show the direction of the gap.
    F4  Evidence status (reference figures C + N6): (a) status of every ESCO link per occupation;
        (b) profile audit of one occupation: every ESCO link's share of ads in H2 2024 (bar) and
        H1 2025 (marker), the occupation's reference levels, and its strongest missing links.
    F6  Why frequency pruning fails (reference figure F): (a) general skills are written less often
        (advertising discount); (b) where the links that frequency pruning would delete end up in ZenESCO.

    F5, F7, F8, F9, F10 need the results of Stages 7-9 and are produced there.

Inputs   BASE\\stage3\\official_edges.parquet, BASE\\stage4\\*, BASE\\stage6\\*
Outputs  BASE\\figures\\F1_architecture.png, F2_official_network.png, F3_digital_gap.png,
         F4_evidence_status.png, F6_frequency_pruning.png   (300 dpi PNG + PDF for the journal)
Run      python figures_stage1_6.py                   # BASE = C:\\skillab_data
         python figures_stage1_6.py C:\\skillab_data "purchaser"    # other occupation for the F4 audit
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
AUDIT_OCC = sys.argv[2] if len(sys.argv) > 2 else "quality engineer"
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- one visual system for the whole paper ----------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#008300", "#6250d6", "#e34948"
ESS, OPT = BLUE, "#9cc3f0"                       # essential = full blue, optional = light blue
STATUS = {"supported": BLUE, "intermediate": "#b9b8b0", "unobservable": "#e6e5e0",
          "weak": ORANGE, "contradicted": RED}
ACTION = {"retain: supported/adequate": "#b9b8b0", "retain: protected": "#7d7c76",
          "retain: no evidence": "#e6e5e0", "retain: implied": "#d4d3cc",
          "DELETE candidate": RED, "DEMOTE candidate": ORANGE, "UPGRADE candidate": BLUE,
          "ADD candidate": GREEN}
CLUSTER_NAME = {1: "Industrial & manufacturing eng.", 2: "Automation & control", 3: "Robotics & mechatronics",
                4: "Maintenance & reliability", 5: "Quality & testing", 6: "Electrical & electronic eng.",
                7: "Product design & development", 8: "CAD & documentation", 9: "Production management",
                10: "Supply chain planning", 11: "Logistics & distribution", 12: "Purchasing"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "figure.dpi": 100})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=SURF)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {name}")


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.5, color=INK, fontweight="bold", pad=6)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def short(t, n=36):
    t = str(t)
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def cluster_order(df, value_col, ascending=False):
    """Occupations ordered by cluster, then by value; returns ordered frame with a y position and gaps."""
    df = df.sort_values(["cluster", value_col], ascending=[True, ascending]).reset_index(drop=True)
    ys, y, last = [], 0.0, None
    for c in df.cluster:
        if last is not None and c != last:
            y += 0.6
        ys.append(y)
        y += 1
        last = c
    df["y"] = ys
    return df


def cluster_labels(ax, df, x):
    for c, g in df.groupby("cluster"):
        ax.text(x, g.y.mean(), CLUSTER_NAME.get(int(c), str(c)), rotation=0, ha="right", va="center",
                fontsize=6.3, color=MUTED, style="italic", transform=ax.get_yaxis_transform())


# ---- data -------------------------------------------------------------------------
E3 = pd.read_parquet(BASE / "stage3" / "official_edges.parquet")
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
OS = pd.read_csv(BASE / "stage4" / "occupation_evidence.csv")
ADD = pd.read_parquet(BASE / "stage6" / "add_candidates.parquet")
E3 = E3[E3.occ_uri.isin(set(EV.occ_uri))]
occ_cluster = EV.groupby("occupation").cluster.first()

# =====================================================================================
# F1 architecture (schematic)
# =====================================================================================
def f1():
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_axes([0, 0, 0.62, 1])
    ax.set_xlim(-0.2, 1.55)
    ax.set_ylim(-0.15, 2.75)
    ax.axis("off")
    sk = 0.42                                                    # perspective skew

    def T(x, y, z):
        return x + sk * y, 0.32 * y + z

    occ = {"o1": (0.12, 0.75), "o2": (0.12, 0.45), "o3": (0.12, 0.15)}
    skl = {f"s{i}": (0.62 + 0.28 * (i % 2), 0.92 - 0.11 * i) for i in range(8)}
    official = [("o1", "s0", "e"), ("o1", "s1", "o"), ("o1", "s2", "o"), ("o2", "s2", "e"), ("o2", "s3", "e"),
                ("o2", "s4", "o"), ("o3", "s5", "e"), ("o3", "s6", "o"), ("o1", "s4", "o")]
    empirical = [("o1", "s0", 0.9), ("o1", "s2", 0.5), ("o1", "s7", 0.8), ("o2", "s2", 0.7), ("o2", "s3", 0.9),
                 ("o3", "s5", 0.8), ("o3", "s6", 0.6), ("o1", "s4", 0.15)]
    updated = {"keep": [("o1", "s0"), ("o2", "s3"), ("o3", "s5"), ("o3", "s6"), ("o2", "s4")],
               "delete": [("o1", "s1")], "upgrade": [("o1", "s2")], "add": [("o1", "s7")],
               "review": [("o1", "s4")], "keep_e": [("o2", "s2")]}
    layers = [(1.9, "Official ESCO graph  G", "essential / optional links"),
              (0.95, "Empirical graph from job ads", "prevalence, lift, observability"),
              (0.0, "Updated ESCO  G′", "Retain · Delete · Upgrade · Add")]
    pos = {**occ, **skl}

    def plane(z, fc):
        pts = [T(0, 0, z), T(1.05, 0, z), T(1.05, 1.02, z), T(0, 1.02, z)]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=MUTED, lw=0.6, zorder=1))

    def node(n, z, alpha=1):
        x, y = T(*pos[n], z)
        if n.startswith("o"):
            ax.scatter(x, y, s=46, marker="s", color=INK2, zorder=5, alpha=alpha, edgecolors="white", linewidths=0.6)
        else:
            ax.scatter(x, y, s=26, marker="o", color="white", edgecolors=INK2, linewidths=0.8, zorder=5, alpha=alpha)

    def edge(a, b, z, **kw):
        (x1, y1), (x2, y2) = T(*pos[a], z), T(*pos[b], z)
        ax.plot([x1, x2], [y1, y2], zorder=3, solid_capstyle="round", **kw)

    for z, title, sub in layers:
        plane(z, "#f7f7f5")
        x, y = T(0, 1.02, z)
        ax.text(x, y + 0.2, title, fontsize=8, color=INK, fontweight="bold", va="bottom")
        ax.text(x, y + 0.08, sub, fontsize=6.6, color=INK2, va="bottom")
    for n in pos:                                                # same node across layers
        xs = [T(*pos[n], z)[0] for z, _, _ in layers]
        ys = [T(*pos[n], z)[1] for z, _, _ in layers]
        ax.plot(xs, ys, color=GRID, lw=0.6, ls=(0, (2, 2)), zorder=2)
    z = layers[0][0]
    for a, b, t in official:
        edge(a, b, z, color=BLUE, lw=1.6 if t == "e" else 0.9, ls="-" if t == "e" else (0, (3, 2)))
    z = layers[1][0]
    for a, b, w in empirical:
        edge(a, b, z, color=AQUA, lw=0.5 + 2.2 * w, alpha=0.85)
    z = layers[2][0]
    for a, b in updated["keep"]:
        edge(a, b, z, color="#9a9993", lw=0.9)
    for a, b in updated["keep_e"]:
        edge(a, b, z, color="#9a9993", lw=1.6)
    for a, b in updated["delete"]:
        edge(a, b, z, color=RED, lw=1.0, ls=(0, (1, 1.5)))
        (x1, y1), (x2, y2) = T(*pos[a], z), T(*pos[b], z)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, "×", color=RED, fontsize=9, ha="center", va="center", zorder=6)
    for a, b in updated["upgrade"]:
        edge(a, b, z, color=BLUE, lw=2.0)
    for a, b in updated["add"]:
        edge(a, b, z, color=GREEN, lw=1.8)
    for a, b in updated["review"]:
        edge(a, b, z, color=YELLOW, lw=1.2, ls=(0, (3, 2)))
    for z, _, _ in layers:
        for n in pos:
            node(n, z)
    legend = [Line2D([], [], marker="s", ls="", color=INK2, label="occupation"),
              Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="skill"),Line2D([], [], color=BLUE, lw=1.6, label="essential"),
              Line2D([], [], color=BLUE, lw=0.9, ls=(0, (3, 2)), label="optional"),
              Line2D([], [], color=AQUA, lw=2.2, label="demand in ads"),
              Line2D([], [], color=RED, lw=1.0, ls=(0, (1, 1.5)), label="delete"),
              Line2D([], [], color=BLUE, lw=2.2, label="upgrade"),
              Line2D([], [], color=GREEN, lw=1.8, label="add"),
              Line2D([], [], color=YELLOW, lw=1.2, ls=(0, (3, 2)), label="expert review")]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-0.05, -0.14), ncol=5, fontsize=6.2,
              handlelength=2.2, columnspacing=1.0)
    # pipeline on the right
    bx = fig.add_axes([0.66, 0.04, 0.33, 0.92])
    bx.axis("off")
    steps = [("Stages 1–2", "Clean job ads; occupation\nfrom title (validated)"),
             ("Stage 3", "Official graph G, protections,\nbaseline structure"),
             ("Stages 4–6", "Evidence per link: demand,\nspecificity, observability;\nimplied-skill gate; candidates"),
             ("Stage 7", "Perturb G; test each edit on\nheld-out ads (H1 2025)"),
             ("Stages 8–9", "Bootstrap stability, joint guard;\nRetain / Delete / Upgrade / Add"),
             ("Stage 10", "Two-phase expert review,\nverified by a domain expert")]
    n = len(steps)
    for i, (h, t) in enumerate(steps):
        y = 1 - (i + 0.5) / n
        bx.add_patch(FancyBboxPatch((0.02, y - 0.065), 0.96, 0.13, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="#f7f7f5", ec=MUTED, lw=0.6, transform=bx.transAxes))
        bx.text(0.07, y + 0.035, h, fontsize=7, fontweight="bold", color=INK, transform=bx.transAxes, va="center")
        bx.text(0.07, y - 0.02, t, fontsize=6.3, color=INK2, transform=bx.transAxes, va="center", linespacing=1.15)
        if i < n - 1:
            bx.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.095), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="<|-", color=MUTED, lw=0.7))
    save(fig, "F1_architecture")


# =====================================================================================
# F2 official network
# =====================================================================================
def f2():
    p = E3.groupby(["occupation", "cluster"]).relation.value_counts().unstack(fill_value=0).reset_index()
    p["k"] = p.essential + p.optional
    p = cluster_order(p, "k", ascending=True)
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.6, wspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    ax.barh(p.y, p.essential, color=ESS, height=0.72, label="essential", edgecolor="white", lw=0.6)
    ax.barh(p.y, p.optional, left=p.essential, color=OPT, height=0.72, label="optional", edgecolor="white", lw=0.6)
    for y, k in zip(p.y, p.k):
        ax.text(k + 4, y, str(k), va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(p.y)
    ax.set_yticklabels(p.occupation, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("skills in the ESCO profile")
    ax.legend(loc="lower right", fontsize=6.6)
    grid_x(ax)
    panel(ax, "a", "Profile size per occupation")
    cluster_labels(ax, p, -0.62)
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    bins = np.logspace(0, np.log10(E3.generality.max() + 1), 22)
    ax.hist([E3.loc[E3.relation == "essential", "generality"], E3.loc[E3.relation == "optional", "generality"]],
            bins=bins, stacked=True, color=[ESS, OPT], edgecolor="white", lw=0.5, label=["essential", "optional"])
    ax.set_xscale("log")
    ax.set_xlabel("ESCO occupations using the skill (log)")
    ax.set_ylabel("links")
    med = E3.generality.median()
    ax.axvline(med, color=INK2, lw=0.7, ls=(0, (2, 2)))
    ax.text(med * 1.1, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", fontsize=6.3, color=INK2)
    panel(ax, "b", "How generic the linked skills are")
    # (c)
    ax = fig.add_subplot(gs[1, 1])
    cats = [("knowledge", E3.skill_type == "knowledge"), ("digital", E3.is_digital), ("green", E3.is_green),
            ("safety /\nlegal", E3.prot_safety), ("transversal", E3.is_transversal), ("language", E3.is_language)]
    x = np.arange(len(cats))
    e = [m[E3.relation == "essential"].mean() * 100 for _, m in cats]
    o = [m[E3.relation == "optional"].mean() * 100 for _, m in cats]
    ax.bar(x - 0.19, e, 0.36, color=ESS, label="essential")
    ax.bar(x + 0.19, o, 0.36, color=OPT, label="optional")
    for xi, v in zip(x, e):
        ax.text(xi - 0.19, v + 0.8, f"{v:.0f}", ha="center", fontsize=5.8, color=INK2)
    for xi, v in zip(x, o):
        ax.text(xi + 0.19, v + 0.8, f"{v:.0f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in cats], fontsize=6.3)
    ax.set_ylabel("% of links")
    panel(ax, "c", "What the links point to")
    save(fig, "F2_official_network")


# =====================================================================================
# F3 digital gap
# =====================================================================================
def f3():
    d = OS.copy()
    d["gap"] = d.digital_specific - d.digital_official
    d = cluster_order(d, "digital_official", ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 0.27 * len(d) + 1.4))
    for _, r in d.iterrows():
        a, b = r.digital_official * 100, r.digital_specific * 100
        col = ORANGE if b > a else BLUE
        if abs(b - a) > 1.0:
            ax.annotate("", xy=(b, r.y), xytext=(a, r.y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, shrinkA=3, shrinkB=3, mutation_scale=8))
        ax.scatter(a, r.y, s=34, color="white", edgecolors=INK2, linewidths=1.0, zorder=4)
        ax.scatter(b, r.y, s=34, color=col, zorder=4)
        ax.text(max(a, b) + 1.2, r.y, f"{b - a:+.0f} pp", va="center", fontsize=6.2, color=col)
    ax.set_yticks(d.y)
    ax.set_yticklabels(d.occupation, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("% digital skills")
    grid_x(ax)
    cluster_labels(ax, d, -0.5)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="ESCO profile"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="ads ask for more digital skills than ESCO lists"),
               Line2D([], [], marker="o", ls="", color=BLUE, label="ESCO lists more digital skills than ads ask for")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1 - 1.2 / (0.27 * len(d) + 1.4)),
              ncol=1, fontsize=6.6)
    ax.set_title("Digital share of the ESCO profile vs of the skills specific to the occupation's ads",
                 loc="left", fontsize=8.5, color=INK, fontweight="bold")
    save(fig, "F3_digital_gap")


# =====================================================================================
# F4 evidence status + profile audit
# =====================================================================================
def f4():
    order = ["supported", "intermediate", "unobservable", "weak", "contradicted"]
    sh = pd.crosstab(EV.occupation, EV.status, normalize="index").reindex(columns=order, fill_value=0)
    sh["cluster"] = sh.index.map(occ_cluster)
    sh["occupation"] = sh.index
    sh["k"] = EV.groupby("occupation").size().reindex(sh.index).values
    sh = cluster_order(sh.reset_index(drop=True), "supported", ascending=False)
    au = EV[EV.occupation == AUDIT_OCC].copy()
    if au.empty:
        print(f"F4: occupation '{AUDIT_OCC}' not found; audit panel skipped")
    seen = au[au.n > 0].sort_values("prev_raw", ascending=True)
    zero = au[au.n == 0]
    adds = ADD[ADD.occupation == AUDIT_OCC].sort_values("prev_raw", ascending=False).head(6).sort_values("prev_raw")
    n_rows = len(seen) + len(adds) + 3
    fig = plt.figure(figsize=(7.2, max(4.8, 0.15 * n_rows + 1.6)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=1.15)
    ax = fig.add_subplot(gs[0, 0])
    left = np.zeros(len(sh))
    for k in order:
        ax.barh(sh.y, sh[k] * 100, left=left, color=STATUS[k], height=0.72, edgecolor="white", lw=0.6, label=k)
        left += sh[k].values * 100
    ax.set_yticks(sh.y)
    ax.set_yticklabels([f"{o} ({k})" for o, k in zip(sh.occupation, sh.k)], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the occupation's ESCO links")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, fontsize=6.3, handlelength=1.2)
    panel(ax, "a", "Evidence status of ESCO links (H2 2024)")
    # (b) audit
    ax = fig.add_subplot(gs[0, 1])
    rows = []
    for _, r in seen.iterrows():
        rows.append(("link", r))
    y = 0
    ys, labels = [], []
    for kind, r in rows:
        ax.barh(y, r.prev_raw * 100, color=STATUS.get(r.status, MUTED), height=0.7, edgecolor="white", lw=0.5)
        if pd.notna(r.prev_raw_h1):
            ax.scatter(r.prev_raw_h1 * 100, y, marker="D", s=10, color=INK, zorder=4, lw=0)
        mk = "●" if r.relation == "essential" else "○"
        labels.append(f"{short(r.skill)}  {mk}")
        ys.append(y)
        y += 1
    if len(zero):
        zc = zero.status.value_counts()
        labels.append(f"+ {len(zero)} links never mentioned")
        ys.append(y)
        ax.text(0.3, y, " · ".join(f"{k} {v}" for k, v in zc.items()), va="center", fontsize=5.8, color=INK2)
        y += 1.6
    if len(adds):
        ax.axhline(y - 0.8, color=GRID, lw=0.8)
        ax.text(0.98, y - 0.55, "not in ESCO: addition candidates", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6, color=GREEN, style="italic")
        for _, r in adds.iterrows():
            ax.barh(y, r.prev_raw * 100, color=GREEN, height=0.7, edgecolor="white", lw=0.5)
            if pd.notna(r.prev_raw_h1):
                ax.scatter(r.prev_raw_h1 * 100, y, marker="D", s=10, color=INK, zorder=4, lw=0)
            labels.append(short(r.skill))
            ys.append(y)
            y += 1
    if len(au):
        n0, rd = au.N_null.iloc[0] * 100, au.R_dem.iloc[0] * 100
        ax.axvline(n0, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
        ax.axvline(rd, color=INK2, lw=0.7, ls=(0, (4, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=5.9)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlabel("% of the occupation's ads mentioning the skill")
    grid_x(ax)
    panel(ax, "b", f"Profile audit: {AUDIT_OCC}")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=3.5, label="H1 2025 (check)"),
                       Line2D([], [], marker="s", ls="", color=GREEN, ms=5, label="not in ESCO"),
                       Line2D([], [], ls="", marker="$●$", color=INK2, label="essential"),
                       Line2D([], [], ls="", marker="$○$", color=INK2, label="optional"),
                       Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="unrelated-skill level"),
                       Line2D([], [], color=INK2, lw=0.8, ls=(0, (4, 2)), label="confirmed-link level")],
              loc="upper center", bbox_to_anchor=(0.35, -0.07), ncol=2, fontsize=6.2)
    save(fig, "F4_evidence_status")


# =====================================================================================
# F6 why frequency pruning fails
# =====================================================================================
def f6():
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    v = EV[EV.visible].copy()
    v["q"] = pd.qcut(v.generality.rank(method="first"), 5, labels=False)
    rng_ = v.groupby("q").generality.agg(["min", "max"])
    x = np.arange(5)
    for j, (rel, col) in enumerate([("essential", ESS), ("optional", OPT)]):
        m = v[v.relation == rel].groupby("q").apply(lambda g: (g.n > 0).mean() * 100).reindex(range(5))
        ax.bar(x + (j - 0.5) * 0.38, m.values, 0.36, color=col, label=rel)
        for xi, val in zip(x, m.values):
            if pd.notna(val):
                ax.text(xi + (j - 0.5) * 0.38, val + 1, f"{val:.0f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(a)}–{int(b)}" for a, b in zip(rng_["min"], rng_["max"])], fontsize=6.3)
    ax.set_xlabel("skill generality (ESCO occupations using it), quintiles")
    ax.set_ylabel("% of ESCO links mentioned at least once")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    panel(ax, "a", "Many ESCO links never appear in ads")
    # (b) flow from frequency pruning to ZenESCO actions
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    naive = EV[EV.n == 0]
    order = ["DELETE candidate", "DEMOTE candidate", "retain: no evidence", "retain: implied", "retain: protected"]
    names = {"DELETE candidate": "delete candidate", "DEMOTE candidate": "demote candidate (ess. → opt.)",
             "retain: no evidence": "kept: absence not informative", "retain: implied": "kept: implied by a related skill",
             "retain: protected": "kept: protected (safety, legal, last link)"}
    cnt = naive.action.value_counts().reindex(order).fillna(0)
    tot = cnt.sum()
    H, gap = 1.0, 0.025
    xl0, xl1, xr0, xr1 = 0.0, 0.07, 0.42, 0.49
    ax.add_patch(Polygon([(xl0, 0), (xl1, 0), (xl1, H), (xl0, H)], fc=INK2, ec="none"))
    ax.text(xl0 - 0.01, H / 2, f"Frequency pruning\ndeletes every link\nnever mentioned\n\n{int(tot):,} links\n"
            f"({tot / len(EV):.0%} of ESCO)", ha="right", va="center", fontsize=6.5, color=INK, linespacing=1.2)
    yl = H
    yr = H + gap * (len(order) - 1)
    scale = H / tot if tot else 0
    for k in order:
        h = cnt[k] * scale
        if h <= 0:
            continue
        col = ACTION[k] if not k.startswith("retain") else {"retain: no evidence": "#cfcec7",
                                                            "retain: implied": "#a9a8a1",
                                                            "retain: protected": "#7d7c76"}[k]
        top_l, bot_l = yl, yl - h
        top_r, bot_r = yr, yr - h
        verts = [(xl1, top_l), (0.25, top_l), (0.25, top_r), (xr0, top_r), (xr0, bot_r), (0.25, bot_r),
                 (0.25, bot_l), (xl1, bot_l), (xl1, top_l)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
                 MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=col, ec="none", alpha=0.35))
        ax.add_patch(Polygon([(xr0, bot_r), (xr1, bot_r), (xr1, top_r), (xr0, top_r)], fc=col, ec="none"))
        ax.text(xr1 + 0.015, (top_r + bot_r) / 2, f"{names[k]}  {int(cnt[k]):,} ({cnt[k] / tot:.0%})",
                va="center", fontsize=6.5, color=INK)
        yl -= h
        yr -= h + gap
    zen = EV.action.isin(["DELETE candidate", "DEMOTE candidate"])
    extra = int((zen & (EV.n > 0)).sum())
    ax.text(xr0, yr - 0.02, f"ZenESCO also flags {extra} links that ARE mentioned but no more than an "
            f"unrelated skill.\nAll candidates are still tested on held-out ads (Stage 7) before any recommendation.",
            fontsize=6, color=INK2, va="top")
    ax.set_xlim(-0.3, 1.25)
    ax.set_ylim(yr - 0.18, H + gap * 5 + 0.05)
    panel(ax, "b", "What happens to the links frequency pruning would delete")
    save(fig, "F6_frequency_pruning")


# =====================================================================================
# F4c network: occupations, their ESCO skills and what the ads say (style of the
# skill-dependency network reference figure: vertical axis = how general a skill is)
# =====================================================================================
NET_OCC = ["production engineer", "automation engineer", "maintenance and repair engineer",
           "quality engineer", "warehouse manager", "purchaser"]


def f_net():
    occs = [o for o in NET_OCC if o in set(EV.occupation)]
    if len(occs) < 2:
        occs = list(EV.groupby("occupation").size().sort_values().index[:6])
    ev_all = EV[EV.occupation.isin(occs)].copy()
    ev = ev_all[ev_all.status.isin(["supported", "weak", "contradicted"])].copy()
    ad = ADD[ADD.occupation.isin(occs)].sort_values("lift", ascending=False).groupby("occupation").head(8).copy()
    eff_f = BASE / "stage7" / "edit_effects.parquet"
    eff = pd.read_parquet(eff_f) if eff_f.exists() else None
    if eff is not None:
        ok = {(o, s_, a_): (bool(i) if a_ in ("ADD candidate", "UPGRADE candidate") else bool(n))
              for o, s_, a_, i, n in zip(eff.occupation, eff.skill_uri, eff.action, eff.improves, eff.noninferior)}
    else:
        ok = None

    def applied(o, s_, act):
        return True if ok is None else ok.get((o, s_, act), False)

    ox = {o: i * 1.0 for i, o in enumerate(occs)}
    links = pd.concat([ev[["occupation", "skill_uri", "skill", "generality", "prev_raw"]].assign(kind=ev.status),
                       ad[["occupation", "skill_uri", "skill", "prev_raw"]].assign(kind="missing")])
    gmap = E3.drop_duplicates("skill_uri").set_index("skill_uri").generality
    links["generality"] = links.skill_uri.map(gmap).fillna(links.generality).fillna(1)
    sk = links.groupby("skill_uri").agg(skill=("skill", "first"), generality=("generality", "first"),
                                        prev=("prev_raw", "max"), n_occ=("occupation", "nunique"),
                                        mx=("occupation", lambda v: np.mean([ox[o] for o in v]))).reset_index()
    sk["y"] = np.log10(sk.generality.clip(lower=1))
    # lane layout: skills of one occupation fan out above it; shared skills sit between their occupations
    offs = np.array([0, -0.3, 0.3, -0.15, 0.15, -0.38, 0.38, -0.22, 0.22, -0.07, 0.07, -0.33, 0.33])
    sk["x"] = sk.mx
    for o in occs:
        own = sk[(sk.n_occ == 1) & (sk.mx == ox[o])].sort_values("y")
        sk.loc[own.index, "x"] = ox[o] + offs[np.arange(len(own)) % len(offs)]
        sk.loc[own.index, "y"] = own.y + (np.arange(len(own)) % 3 - 1) * 0.045
    r_ = np.random.default_rng(3)
    shared = sk.n_occ > 1
    sk.loc[shared, "x"] += r_.uniform(-0.12, 0.12, shared.sum())
    sk["size"] = 5 + 110 * np.sqrt(sk.prev.fillna(0))
    rank_ = {"missing": 0, "supported": 1, "contradicted": 2, "weak": 3}
    best = links.assign(r=links.kind.map(rank_)).sort_values("r").groupby("skill_uri").kind.first()
    sk["kind"] = sk.skill_uri.map(best)
    P = sk.set_index("skill_uri")
    OY = -0.55
    KC = {"supported": BLUE, "weak": ORANGE, "contradicted": RED, "missing": GREEN}

    def curve(ax, x0, y0, x1, y1, col, lw, ls="-", alpha=0.5, z=2):
        verts = [(x0, y0), (x0, y0 + 0.35 * (y1 - y0)), (x1, y1 - 0.45 * (y1 - y0)), (x1, y1)]
        ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                               fc="none", ec=col, lw=lw, ls=ls, alpha=alpha, zorder=z, capstyle="round"))

    def draw(ax, after, labels):
        keep = []
        order = {"supported": 0, "contradicted": 1, "weak": 2}
        for _, r in ev.sort_values("status", key=lambda c: c.map(order)).iterrows():
            p_ = P.loc[r.skill_uri]
            col, lw, al, ls = KC[r.status], 0.35 + 2.2 * np.sqrt(r.prev_raw), 0.45, "-"
            if r.relation == "optional":
                ls = (0, (2.5, 1.2))
            if after:
                if r.action == "DELETE candidate" and applied(r.occupation, r.skill_uri, r.action):
                    continue
                if r.action == "UPGRADE candidate" and applied(r.occupation, r.skill_uri, r.action):
                    ls, lw, al = "-", lw + 0.9, 0.85
                if r.status != "supported":
                    col, al = "#b9b8b0", 0.35
            curve(ax, ox[r.occupation], OY, p_.x, p_.y, col, lw, ls, al)
            keep.append(r.skill_uri)
        for _, r in ad.iterrows():
            if after and not applied(r.occupation, r.skill_uri, "ADD candidate"):
                continue
            p_ = P.loc[r.skill_uri]
            curve(ax, ox[r.occupation], OY, p_.x, p_.y, GREEN, 0.5 + 2.2 * np.sqrt(r.prev_raw),
                  "-" if after else (0, (1, 1.1)), 0.8, z=3)
            keep.append(r.skill_uri)
        q = P.loc[list(dict.fromkeys(keep))]
        cols = [KC[k] if not (after and k in ("weak", "contradicted")) else "#cfcec7" for k in q.kind]
        ax.scatter(q.x, q.y, s=q["size"], c=cols, edgecolors="white", linewidths=0.5, zorder=5)
        for o in occs:
            ax.scatter(ox[o], OY, s=150, marker="s", c=INK, zorder=6, edgecolors="white", linewidths=1)
            nm = o.replace(" and repair", " & repair").replace("production engineering technician", "production eng. technician")
            ax.text(ox[o], OY - 0.14, nm.replace(" ", "\n", 1) if len(nm) > 16 else nm, ha="center", va="top",
                    fontsize=6.2, color=INK, fontweight="bold", linespacing=1.05)
            if not after:
                c = ev_all[ev_all.occupation == o].status.value_counts()
                ax.text(ox[o], OY - 0.42, f"{c.get('supported', 0)} supported · {c.get('weak', 0) + c.get('contradicted', 0)} weak\n"
                        f"{c.get('unobservable', 0) + c.get('intermediate', 0)} no clear evidence (not drawn)",
                        ha="center", va="top", fontsize=5.3, color=MUTED, linespacing=1.1)
        for i, s_ in enumerate(labels):
            p_ = P.loc[s_]
            col = KC[p_.kind]
            if after and p_.kind in ("weak", "contradicted"):
                continue
            dx = 10 if p_.x >= np.mean(list(ox.values())) else -10
            ax.annotate(short(p_.skill, 28), xy=(p_.x, p_.y), xytext=(dx, 6 + 5 * (i % 2)), textcoords="offset points",
                        fontsize=5.5, color=col, ha="left" if dx > 0 else "right", zorder=8,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.4, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(-0.6, len(occs) - 0.4)
        ax.set_ylim(OY - (0.75 if not after else 0.45), np.log10(P.generality.max()) + 0.35)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        yt = [1, 3, 10, 30, 100, 300]
        ax.set_yticks([np.log10(v) for v in yt])
        ax.set_yticklabels([str(v) for v in yt])
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)

    lab = []
    for o in occs:
        e = ev[ev.occupation == o]
        lab += list(e[e.status == "supported"].nlargest(1, "prev_raw").skill_uri)
        lab += list(e[e.status.isin(["weak", "contradicted"])].nlargest(1, "generality").skill_uri)
        lab += list(ad[ad.occupation == o].nlargest(1, "prev_raw").skill_uri)
    lab = list(dict.fromkeys(lab))
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.8], hspace=0.42, wspace=0.75)
    ax = fig.add_subplot(gs[0, :])
    draw(ax, False, lab)
    ax.set_ylabel("skill generality\n(ESCO occupations using the skill, log)")
    panel(ax, "a", "What the job ads say about the ESCO links of six occupations (H2 2024)")
    handles = [Line2D([], [], color=BLUE, lw=2, label="supported"),
               Line2D([], [], color=ORANGE, lw=2, label="weak"),
               Line2D([], [], color=RED, lw=2, label="contradicted (strong elsewhere)"),
               Line2D([], [], color=GREEN, lw=2, ls=(0, (1, 1.1)), label="missing: not in ESCO"),
               Line2D([], [], color=INK2, lw=1, label="essential"),
               Line2D([], [], color=INK2, lw=1, ls=(0, (2.5, 1.2)), label="optional"),
               Line2D([], [], marker="o", ls="", color=MUTED, ms=6, label="node size = share of ads")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=7, fontsize=5.8,
              handlelength=1.6, columnspacing=0.9)
    bx = fig.add_subplot(gs[1, :2])
    draw(bx, True, [])
    bx.set_ylabel("generality (log)")
    panel(bx, "b", "After ZenESCO")
    src = ("edits kept by the held-out test (Stage 7)" if eff is not None else "candidate edits (before Stage 7)")
    bx.text(1.0, 1.01, src, transform=bx.transAxes, fontsize=5.8, color=MUTED, ha="right", va="bottom")
    cx = fig.add_subplot(gs[1, 2])
    rows = []
    for o in occs:
        e = EV[EV.occupation == o]
        rows.append((o,
                     -sum(applied(o, s_, a_) for s_, a_ in zip(e.skill_uri, e.action) if a_ == "DELETE candidate"),
                     sum(applied(o, s_, a_) for s_, a_ in zip(e.skill_uri, e.action) if a_ == "UPGRADE candidate"),
                     sum(applied(o, s_, "ADD candidate") for s_ in ADD[ADD.occupation == o].skill_uri)))
    cnt = pd.DataFrame(rows, columns=["o", "del", "up", "add"])
    yy = np.arange(len(cnt))
    cx.barh(yy, cnt["del"], color=RED, height=0.6, label="delete")
    cx.barh(yy, cnt["add"], color=GREEN, height=0.6, label="add")
    cx.barh(yy, cnt["up"], left=cnt["add"], color=BLUE, height=0.6, label="upgrade")
    for y_, d_, a_, u_ in zip(yy, cnt["del"], cnt["add"], cnt["up"]):
        if d_:
            cx.text(d_ - 0.3, y_, str(-d_), ha="right", va="center", fontsize=5.6, color=RED)
        if a_ + u_:
            cx.text(a_ + u_ + 0.3, y_, f"+{a_ + u_}", ha="left", va="center", fontsize=5.6, color=GREEN)
    cx.axvline(0, color=INK2, lw=0.6)
    cx.set_yticks(yy)
    cx.set_yticklabels([short(o, 20) for o in cnt.o], fontsize=6)
    cx.invert_yaxis()
    lim = max(1, cnt["del"].abs().max(), (cnt["add"] + cnt["up"]).max()) * 1.35
    cx.set_xlim(-lim, lim)
    cx.set_xlabel("links changed")
    cx.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1)
    panel(cx, "c", "Changes")
    save(fig, "F4c_network")

for f in (f1, f2, f3, f4, f6):   # f_net (old network figure) replaced by figures_results.py
    try:
        f()
    except Exception as ex:
        print(f"{f.__name__} failed: {ex!r}")
print(f"Figures in {OUT}")
