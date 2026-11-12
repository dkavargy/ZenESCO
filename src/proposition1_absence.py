"""
ZenESCO - Proposition 1: when is the absence of a skill from job ads informative?
================================================================================

Setting (Eq. 4 of the paper; Stages 4-5)
    Occupation o has A_o advertisements in the selection window. If skill s were required for o, it would be
    written in an advertisement with probability q = delta_g * R_dem(o)   (reference level times the advertising
    discount of the skill's generality group). The number of advertisements mentioning s is then X ~ Bin(A_o, q).
    The absence (or rarity) of s is INFORMATIVE - the link may be judged weak - only if observing at most n_os
    mentions would be unlikely for a required skill:    F(n_os; A_o, q) = Pr(X <= n_os) < a   (a = 0.05).

Proposition 1
    (i)  F(n; A, q) = I_{1-q}(A - n, n + 1) (regularised incomplete beta function) and F is strictly decreasing in
         A and in q. Hence for every n and q there is a smallest number of advertisements A*(n, q) from which on
         the absence is informative.
    (ii) For a skill never mentioned (n = 0):   A*(0, q) = floor( ln a / ln(1 - q) ) + 1,   and
                 (1 - q) ln(1/a) / q  <  A*(0, q)  <=  ln(1/a) / q + 1.
         With a = 0.05, ln(1/a) = 3.00: a "rule of three" - absence is informative only if at least about three
         mentions would be expected from a required skill (A_o * q >= 3).
Proof
    The identity in (i) is the standard relation between the binomial and the beta distribution. Bin(A, q) is
    stochastically increasing in A and in q, so F(n; A, q) decreases in both. For n = 0, F = (1 - q)^A < a
    <=> A > ln a / ln(1 - q), because ln(1 - q) < 0; the smallest such integer is floor(.) + 1. The bounds follow
    from q <= -ln(1 - q) <= q / (1 - q) for 0 < q < 1.
Corollary
    Frequency pruning (delete every link with n_os = 0) is a valid test only for links with A_o >= A*(0, q). For
    the others the data are consistent with the link being required, so their deletion is not supported by evidence.

What this script does
    1. verifies (i) and (ii) numerically on a dense grid (exact binomial computations);
    2. applies the proposition to every official link of the analysed occupations (Stage 6 output), checks that it
       reproduces Stage 5's power test, and counts the never-mentioned links whose absence is not informative;
    3. writes a figure (A* against q with every occupation x generality group) and sentence-ready numbers.
Inputs   BASE\\stage6\\screened_links.parquet  (columns n, N, expected_level, relation, occupation, ...)
Outputs  BASE\\stage5b\\proposition1_links.csv, proposition1_occupations.csv, F_prop1_absence.png/pdf,
         proposition1_report.txt
Run      python proposition1_absence.py                 # BASE = C:\\skillab_data
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, special

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
OUT = BASE / "stage5b"
A_LEVEL = 0.05                       # ALPHA_POW of Stages 4-5
T0 = time.time()
REPORT = []


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def a_star0(q, a=A_LEVEL):
    """smallest A with (1-q)^A < a  (n = 0), closed form"""
    q = np.asarray(q, dtype=float)
    return np.floor(np.log(a) / np.log1p(-q)) + 1


def a_star(n, q, a=A_LEVEL, a_max=10 ** 6):
    """smallest A >= n with F(n; A, q) < a, by bisection on the monotone binomial CDF"""
    lo, hi = max(n, 1), max(n, 1)
    while stats.binom.cdf(n, hi, q) >= a:
        hi *= 2
        if hi > a_max:
            return np.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if stats.binom.cdf(n, mid, q) < a:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ============================================================
# 1. NUMERICAL VERIFICATION OF THE PROPOSITION
# ============================================================
say("ZenESCO - Proposition 1 report: when is the absence of a skill from job ads informative?")
say("=" * 84)
qs = np.unique(np.concatenate([np.geomspace(1e-3, 0.9, 400), np.linspace(0.01, 0.6, 60)]))
L = np.log(1 / A_LEVEL)
ok_closed = ok_bounds = ok_beta = ok_mono = True
for q in qs:
    A0 = int(a_star0(q))
    if not (A0 == a_star(0, q)):
        ok_closed = False
    if not ((1 - q) * L / q < A0 <= L / q + 1):
        ok_bounds = False
for n in [0, 1, 2, 5]:
    for q in [0.01, 0.05, 0.2]:
        A = np.arange(max(n, 1), 600)
        F = stats.binom.cdf(n, A, q)
        Fb = special.betainc(A - n, n + 1, 1 - q) if n < A.min() else None
        if Fb is not None and not np.allclose(F, Fb, atol=1e-12):
            ok_beta = False
        if not np.all(np.diff(F) <= 1e-15):
            ok_mono = False
    A = 200
    Fq = stats.binom.cdf(n, A, np.linspace(0.001, 0.5, 500))
    if not np.all(np.diff(Fq) <= 1e-15):
        ok_mono = False
say(f"Verification on {len(qs)} values of q in [0.001, 0.9], a = {A_LEVEL}:")
say(f"  closed form A*(0,q) = floor(ln a / ln(1-q)) + 1 equals the exact search: {ok_closed}")
say(f"  bounds (1-q) ln(1/a)/q < A*(0,q) <= ln(1/a)/q + 1 hold:                  {ok_bounds}")
say(f"  F(n; A, q) = I_(1-q)(A-n, n+1) (beta identity):                          {ok_beta}")
say(f"  F decreasing in A and in q:                                              {ok_mono}")
if not (ok_closed and ok_bounds and ok_beta and ok_mono):
    say("  *** a check failed - do not use the proposition before inspecting the report")
say("")
say("A*(n, q): smallest number of advertisements from which n or fewer mentions are informative (a = 0.05)")
tq = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
say("  q      " + "".join(f"{q:>8.2f}" for q in tq))
for n in [0, 1, 2, 5]:
    say(f"  n = {n}  " + "".join(f"{a_star(n, q):>8.0f}" for q in tq))
say(f"  rule of three: ln(1/a) = {L:.3f}, so A*(0,q) ~ 3/q")

# ============================================================
# 2. APPLICATION TO THE OFFICIAL LINKS
# ============================================================
EV = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
need = {"n", "N", "expected_level", "occupation"}
if not need.issubset(EV.columns):
    sys.exit(f"screened_links.parquet lacks {sorted(need - set(EV.columns))} - run Stages 4 and 5-6 first")
d = EV.copy()
d["q"] = d.expected_level.astype(float)
d = d[d.q.notna() & (d.q > 0) & (d.q < 1)].copy()
d["A"] = d.N.astype(int)
d["A_star0"] = a_star0(d.q).astype(int)
d["A_star_n"] = [a_star(int(n), q) for n, q in zip(d.n, d.q)]
d["informative"] = d.A >= d.A_star_n
d["expected_mentions"] = d.A * d.q
if "powered_adj" in d.columns:
    agree = (d.informative == d.powered_adj.astype(bool)).mean()
    say("")
    say(f"Check against Stage 5 (power test after the advertising discount): agreement {agree:.1%} of {len(d):,} links")
zero = d[d.n == 0]
un = zero[~zero.informative]
say("")
say(f"Official links with a reference level: {len(d):,} (of {len(EV):,}); never mentioned in H2 2024 ads: {len(zero):,}")
say(f"  q = delta_g * R_dem(o) ranges from {d.q.min():.3f} to {d.q.max():.3f} (median {d.q.median():.3f}); "
    f"A*(0, q) from {d.A_star0.min():,} to {d.A_star0.max():,} advertisements (median {int(d.A_star0.median()):,})")
say(f"  advertisements per occupation A_o: {d.groupby('occupation').A.first().min():,} to "
    f"{d.groupby('occupation').A.first().max():,}")
say(f"  never-mentioned links whose absence is NOT informative (A_o < A*): {len(un):,} of {len(zero):,} "
    f"({len(un) / max(len(zero), 1):.1%}); frequency pruning would delete all of them without evidence")
if "relation" in d.columns:
    say(f"    of which essential: {(un.relation == 'essential').sum():,}, optional: {(un.relation == 'optional').sum():,}")
if "gen_group" in d.columns:
    g = zero.groupby("gen_group").agg(links=("q", "size"), q=("q", "median"), A_star=("A_star0", "median"),
                                      not_informative=("informative", lambda x: (~x).mean()))
    say("  by generality group of the skill (1 = most specific):")
    say("  " + g.round(3).to_string().replace("\n", "\n  "))
occ = (d.assign(zero=d.n == 0, zero_uninf=(d.n == 0) & ~d.informative)
       .groupby("occupation").agg(A_o=("A", "first"), q_max=("q", "max"), q_min=("q", "min"),
                                  A_star_max=("A_star0", "max"), never_mentioned=("zero", "sum"),
                                  not_informative=("zero_uninf", "sum")).reset_index())
occ["share_not_informative"] = occ.not_informative / occ.never_mentioned.clip(lower=1)
say("")
say("Per occupation (q_max = level of the most specific skills, q_min = of the most general):")
say(occ.sort_values("A_o").round(3).to_string(index=False))

OUT.mkdir(parents=True, exist_ok=True)
cols = [c for c in ["occupation", "skill", "skill_uri", "relation", "n", "A", "q", "expected_mentions", "A_star0",
                    "A_star_n", "informative", "status", "action"] if c in d.columns]
d[cols].to_csv(OUT / "proposition1_links.csv", index=False, encoding="utf-8-sig")
occ.to_csv(OUT / "proposition1_occupations.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 3. FIGURE
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    RED, GREY, BLUE, ORANGE = "#e34948", "#c9c8c2", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    qq = np.geomspace(max(d.q.min() * 0.7, 1e-3), min(d.q.max() * 1.4, 0.9), 300)
    ax.fill_between(qq, (1 - qq) * L / qq, L / qq + 1, color=GREY, alpha=0.35, lw=0, label="bounds of Proposition 1")
    ax.plot(qq, a_star0(qq), color=INK2, lw=1.2, label="A*(0, q): no mention")
    ax.plot(qq, [a_star(1, q) for q in qq], color=INK2, lw=0.8, ls="--", label="A*(1, q): one mention")
    pts = (d[d.n == 0].groupby(["occupation", "q"]).agg(A=("A", "first"), k=("A", "size"),
                                                        inf=("informative", "first")).reset_index())
    ax.scatter(pts.q, pts.A, s=6 + 2.5 * pts.k, c=np.where(pts.inf, RED, BLUE), alpha=0.75, lw=0, zorder=3)
    ax.scatter([], [], c=RED, s=18, label="absence informative")
    ax.scatter([], [], c=BLUE, s=18, label="absence not informative")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("expected share of ads mentioning a required skill, q = \u03b4_g \u00b7 R_dem(o)")
    ax.set_ylabel("advertisements of the occupation, A\u2092")
    ax.legend(fontsize=5.8, loc="lower left")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.3, "(a) Informativeness of absence (\u03b1 = 0.05)", transform=ax.transAxes, ha="center",
            fontsize=8, color=INK)
    o2 = occ.sort_values("never_mentioned")
    y = np.arange(len(o2))
    bx.barh(y, o2.never_mentioned - o2.not_informative, color=RED, height=0.65, label="informative", zorder=3)
    bx.barh(y, o2.not_informative, left=o2.never_mentioned - o2.not_informative, color=BLUE, height=0.65,
            label="not informative", zorder=3)
    bx.set_yticks(y)
    bx.set_yticklabels([f"{o} ({a:,})" for o, a in zip(o2.occupation, o2.A_o)], fontsize=5.8)
    bx.set_xlabel("official links never mentioned in H2 2024 ads")
    bx.legend(fontsize=5.8, loc="lower right")
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    bx.text(0.35, -0.3, "(b) Never-mentioned links per occupation (A\u2092)", transform=bx.transAxes, ha="center",
            fontsize=8, color=INK)
    fig.tight_layout(w_pad=1.5)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"F_prop1_absence.{ext}", dpi=300, bbox_inches="tight")
    say("\nFigure saved: F_prop1_absence.png")
except Exception as ex:
    import traceback
    traceback.print_exc()
    say(f"Figure skipped: {ex!r}")

say(f"Runtime: {time.time() - T0:.0f}s")
(OUT / "proposition1_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
