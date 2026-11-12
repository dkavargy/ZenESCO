"""
ZenESCO - Stage 7: graph perturbation and held-out utility
==========================================================

Question
    What happens to ESCO's usefulness when links are deleted, demoted, upgraded or added - and do
    ZenESCO's evidence-based edits keep or improve it where other ways of editing damage it?

Utility, measured on HELD-OUT ads (H1 2025, never used to choose edits; reweighted to H2's portal mix)
    U1 Identification  each ad of an analysed occupation is scored against every analysed occupation's
                       ESCO profile (IDF-weighted cosine between the ad's skills and the profile;
                       essential = 1, optional = 0.5). Accuracy = the true occupation ranks first;
                       MRR = mean reciprocal rank of the true occupation.
    U2 Demand coverage share of skill mentions in an occupation's ads that its profile contains (mean over occupations)
    U3 Distinctiveness median over occupations of 1 - max cosine similarity to another analysed occupation

A. Perturbation curves (Figure F5, reference figures A / N10 / N12)
    Deletion of 0-50% of the deletable links (protected links are never deleted by any strategy):
      random                       uniform random order (20 repetitions, mean and 95% band)
      frequency pruning            least mentioned in H2 ads first (ties random; 20 repetitions)
      generic first (structure)    most widely used ESCO skills first
      ZenESCO                      Stage 6 delete and demote candidates, strongest evidence first (stops at its budget)
      positive control             links the ads CONFIRM, most specific first (should hurt)
    Additions of 0-10 skills per occupation:
      ZenESCO                      Stage 6 add candidates, most specific first
      frequency                    most mentioned skills not in the profile (no specificity test)
      random                       random skills seen in the occupation's ads (20 repetitions)
    Bands for deterministic strategies: 95% bootstrap over held-out ads (200 resamples).

B. Effect of every single candidate edit (input to the Stage 9 decisions)
    dU = change in MRR (all held-out ads), 90% bootstrap interval; dDC_o and dDist_o for the edited occupation.
    Non-inferiority margin m_U = 5th percentile of dU over 300 random single deletions (the noise of the test).

Inputs   BASE\\stage1\\ads_clean.parquet, ad_skills_clean.parquet; BASE\\stage2\\ad_occupation.parquet
         BASE\\stage6\\screened_links.parquet, add_candidates.parquet; BASE\\stage4\\pair_evidence.parquet
         BASE\\esco\\occupationSkillRelations_en.csv (skill generality for IDF)
Outputs  BASE\\stage7\\curves.csv, edit_effects.parquet, F5_utility_under_editing.png/pdf, stage7_report.txt
Run      python stage7_perturbation.py              # BASE = C:\\skillab_data
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
S1, S2, S4, S6, ESCO = BASE / "stage1", BASE / "stage2", BASE / "stage4", BASE / "stage6", BASE / "esco"
OUT = BASE / "stage7"
WIN_SEL, WIN_TEST = "H2_2024", "H1_2025"
ALPHA = 0.5
FRACS = np.round(np.arange(0, 0.51, 0.05), 2)
ADD_K = list(range(0, 11))
REPS = 20
BOOT = 200
SEED = 20261026
rng = np.random.default_rng(SEED)
T0 = time.time()
REPORT = []


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


# ============================================================
# 1. DATA
# ============================================================
log("Loading")
EV = pd.read_parquet(S6 / "screened_links.parquet")
ADD = pd.read_parquet(S6 / "add_candidates.parquet")
PE = pd.read_parquet(S4 / "pair_evidence.parquet")
OCC = sorted(EV.occ_uri.unique())
oi = {o: i for i, o in enumerate(OCC)}
olab = EV.groupby("occ_uri").occupation.first().to_dict()

ads = pd.read_parquet(S1 / "ads_clean.parquet", columns=["ad_id", "portal", "window", "analysable"])
ads = ads[ads.analysable & ads.window.isin([WIN_SEL, WIN_TEST])]
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(S2 / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
ads = ads.merge(occ_of, on="ad_id", how="left")
ads = ads[ads.occ_uri.isin(set(OCC))]
# portal reweighting: H1 test ads get H2's portal mix
mix2 = ads[ads.window == WIN_SEL].portal.value_counts(normalize=True)
test = ads[ads.window == WIN_TEST].reset_index(drop=True)
mix1 = test.portal.value_counts(normalize=True)
test["w"] = test.portal.map(lambda p: mix2.get(p, 0) / mix1[p]).fillna(0)
test.loc[test.w == 0, "w"] = 1e-6
ask = pd.read_parquet(S1 / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
ask = ask[ask.ad_id.isin(set(test.ad_id))].drop_duplicates()

rel_f = [p for p in ESCO.glob("*.csv") if p.name.endswith("occupationSkillRelations_en.csv")][0]
rel = pd.read_csv(rel_f, dtype=str, usecols=["occupationUri", "skillUri"])
N_OCC = rel.occupationUri.nunique()
gen = rel.groupby("skillUri").occupationUri.nunique()

SK = sorted(set(EV.skill_uri) | set(ADD.skill_uri) | set(ask.skill_uri) | set(PE.skill_uri))
si = {s: i for i, s in enumerate(SK)}
idf = np.array([np.log(N_OCC / gen.get(s, 1)) for s in SK])
idf = np.maximum(idf, 0.05)
aid = {a: i for i, a in enumerate(test.ad_id)}
X = sparse.csr_matrix((np.ones(len(ask)), (ask.ad_id.map(aid).values, ask.skill_uri.map(si).values)),
                      shape=(len(test), len(SK)))
X.data[:] = 1.0
true = test.occ_uri.map(oi).values
W = test.w.values
n_mentions = np.asarray(X.sum(axis=1)).ravel()

W0 = np.zeros((len(OCC), len(SK)))
for o, s, r in zip(EV.occ_uri, EV.skill_uri, EV.relation):
    W0[oi[o], si[s]] = 1.0 if r == "essential" else ALPHA

say("ZenESCO - Stage 7 report: graph perturbation and held-out utility")
say("=" * 70)
say(f"Occupations {len(OCC)} | official links {int((W0 > 0).sum()):,} | held-out ads ({WIN_TEST}) {len(test):,} "
    f"with {len(ask):,} skill mentions; portal weights to the {WIN_SEL} mix")


# ============================================================
# 2. UTILITY
# ============================================================
def per_ad(Wm):
    V = Wm * idf
    nrm = np.sqrt((V ** 2).sum(axis=1))
    Vn = V / np.maximum(nrm, 1e-12)[:, None]
    S = np.asarray(X @ Vn.T)
    ts = S[np.arange(len(S)), true]
    higher = (S > ts[:, None]).sum(axis=1)
    ties = (S == ts[:, None]).sum(axis=1) - 1
    rank = 1 + higher + 0.5 * ties
    acc = ((higher == 0) & (ties == 0) & (ts > 0)).astype(float)
    rr = 1.0 / rank
    cov = np.asarray((X.multiply((Wm[true] > 0))).sum(axis=1)).ravel()
    return acc, rr, cov, Vn


def dist_med(Vn):
    C = Vn @ Vn.T
    np.fill_diagonal(C, -1)
    return float(np.median(1 - C.max(axis=1)))


def summary(acc, rr, cov, idx=None):
    if idx is None:
        idx = np.arange(len(acc))
    w = W[idx]
    a = np.sum(w * acc[idx]) / w.sum()
    m = np.sum(w * rr[idx]) / w.sum()
    t = true[idx]
    dc = np.mean([cov[idx][t == k].sum() / max(n_mentions[idx][t == k].sum(), 1) for k in range(len(OCC))])
    return a, m, dc


BIDX = rng.integers(0, len(test), size=(BOOT, len(test)))


def evaluate(Wm, boot=False):
    acc, rr, cov, Vn = per_ad(Wm)
    a, m, dc = summary(acc, rr, cov)
    out = {"acc": a, "mrr": m, "dc": dc, "dist": dist_med(Vn)}
    if boot:
        bs = np.array([summary(acc, rr, cov, b) for b in BIDX])
        for j, k in enumerate(["acc", "mrr", "dc"]):
            out[k + "_lo"], out[k + "_hi"] = np.percentile(bs[:, j], [2.5, 97.5])
    return out


base = evaluate(W0, boot=True)
say(f"Original ESCO: identification accuracy {base['acc']:.3f} [{base['acc_lo']:.3f}, {base['acc_hi']:.3f}], "
    f"MRR {base['mrr']:.3f}, demand coverage {base['dc']:.3f}, distinctiveness {base['dist']:.3f} "
    f"(chance accuracy {1 / len(OCC):.3f})")

# ============================================================
# 3. DELETION CURVES
# ============================================================
log("Deletion curves")
pool = EV[~EV.protected_any].copy()
pool["oi"], pool["si"] = pool.occ_uri.map(oi), pool.skill_uri.map(si)
NP = len(pool)


def apply_del(order_rows, k, weights=None):
    Wm = W0.copy()
    sub = order_rows[:k]
    vals = 0.0 if weights is None else weights[:k]
    Wm[sub.oi.values, sub.si.values] = vals
    return Wm


zen = pool[pool.action.isin(["DELETE candidate", "DEMOTE candidate"])].copy()
zen["key"] = -zen.pr_below_null + zen.lift.rank(pct=True) * 1e-3
zen = zen.sort_values("key")
zen_w = np.where(zen.action == "DEMOTE candidate", ALPHA, 0.0)
ctrl = pool[pool.status == "supported"].sort_values("lift", ascending=False)
generic = pool.sort_values("generality", ascending=False)

rows = []
for f in FRACS:
    k = int(round(f * NP))
    for rep in range(REPS):
        r1 = pool.sample(frac=1, random_state=SEED + rep)
        e = evaluate(apply_del(r1, k))
        rows.append({"panel": "delete", "strategy": "random", "x": f, "rep": rep, **e})
        r2 = pool.assign(t=np.random.default_rng(SEED + rep).random(NP)).sort_values(["prev_raw", "t"])
        e = evaluate(apply_del(r2, k))
        rows.append({"panel": "delete", "strategy": "frequency pruning", "x": f, "rep": rep, **e})
    rows.append({"panel": "delete", "strategy": "generic first", "x": f, "rep": 0,
                 **evaluate(apply_del(generic, k), boot=True)})
    if k <= len(zen):
        rows.append({"panel": "delete", "strategy": "ZenESCO", "x": f, "rep": 0,
                     **evaluate(apply_del(zen, k, zen_w), boot=True)})
    if k <= len(ctrl):
        rows.append({"panel": "delete", "strategy": "positive control", "x": f, "rep": 0,
                     **evaluate(apply_del(ctrl, k), boot=True)})
# ZenESCO's full budget as its own point
fz = len(zen) / NP
rows.append({"panel": "delete", "strategy": "ZenESCO", "x": round(fz, 4), "rep": 0,
             **evaluate(apply_del(zen, len(zen), zen_w), boot=True)})

# ============================================================
# 4. ADDITION CURVES
# ============================================================
log("Addition curves")
ADD["oi"], ADD["si"] = ADD.occ_uri.map(oi), ADD.skill_uri.map(si)
ADD["w_new"] = np.where(ADD.proposed_type == "essential", 1.0, ALPHA)
pe = PE[PE.occ_uri.isin(set(OCC)) & (PE.relation == "")].copy()
pe["oi"], pe["si"] = pe.occ_uri.map(oi), pe.skill_uri.map(si)


def apply_add(table, k, col_w=None):
    Wm = W0.copy()
    for o, g in table.groupby("oi"):
        g = g.head(k)
        Wm[o, g.si.values] = ALPHA if col_w is None else g[col_w].values
    return Wm


zen_add = ADD.sort_values("lift", ascending=False)
freq_add = pe.sort_values("n", ascending=False)
for k in ADD_K:
    rows.append({"panel": "add", "strategy": "ZenESCO", "x": k, "rep": 0,
                 **evaluate(apply_add(zen_add, k, "w_new"), boot=True)})
    rows.append({"panel": "add", "strategy": "most frequent (no specificity)", "x": k, "rep": 0,
                 **evaluate(apply_add(freq_add, k), boot=True)})
    for rep in range(REPS):
        rnd = pe[pe.n > 0].sample(frac=1, random_state=SEED + rep)
        rows.append({"panel": "add", "strategy": "random", "x": k, "rep": rep, **evaluate(apply_add(rnd, k))})
CV = pd.DataFrame(rows)

# ============================================================
# 5. SINGLE-EDIT EFFECTS
# ============================================================
log("Single-edit effects")
acc0, rr0, cov0, Vn0 = per_ad(W0)
V0 = W0 * idf
nrm0 = np.sqrt((V0 ** 2).sum(axis=1))
S0raw = np.asarray(X @ V0.T)
Xc = X.tocsc()
own_tot = np.array([n_mentions[true == k].sum() for k in range(len(OCC))])


def single(o, s, w_new):
    w_old = W0[o, s]
    if w_old == w_new:
        return None
    col = Xc[:, s].toarray().ravel()
    raw = S0raw[:, o] + (w_new - w_old) * idf[s] * col
    nrm = np.sqrt(max(nrm0[o] ** 2 - (w_old * idf[s]) ** 2 + (w_new * idf[s]) ** 2, 1e-24))
    S = S0raw / np.maximum(nrm0, 1e-12)
    S[:, o] = raw / nrm
    ts = S[np.arange(len(S)), true]
    rank = 1 + (S > ts[:, None]).sum(axis=1) + 0.5 * ((S == ts[:, None]).sum(axis=1) - 1)
    d = 1.0 / rank - rr0
    dU = np.sum(W * d) / W.sum()
    bs = (W[BIDX] * d[BIDX]).sum(axis=1) / W[BIDX].sum(axis=1)
    mo = true == o
    dDC = (int(w_new > 0) - int(w_old > 0)) * col[mo].sum() / max(own_tot[o], 1)
    v = V0[o].copy()
    v[s] = w_new * idf[s]
    vn = v / max(np.linalg.norm(v), 1e-12)
    sims = Vn0 @ vn
    sims[o] = -1
    d_before = 1 - np.max(np.delete(Vn0 @ Vn0[o], o))
    dDist = (1 - sims.max()) - d_before
    return dU, np.percentile(bs, 5), np.percentile(bs, 95), dDC, dDist


# noise margin from random single deletions
rand_d = []
for _, r in pool.sample(min(300, NP), random_state=SEED).iterrows():
    res = single(r.oi, r.si, 0.0)
    if res:
        rand_d.append(res[0])
m_U = float(np.percentile(rand_d, 5)) if rand_d else 0.0

eff = []
edits = []
for _, r in EV[EV.action.isin(["DELETE candidate", "DEMOTE candidate", "UPGRADE candidate"])].iterrows():
    w_new = {"DELETE candidate": 0.0, "DEMOTE candidate": ALPHA, "UPGRADE candidate": 1.0}[r.action]
    edits.append((r.occ_uri, r.skill_uri, r.action, oi[r.occ_uri], si[r.skill_uri], w_new))
for _, r in ADD.iterrows():
    edits.append((r.occ_uri, r.skill_uri, "ADD candidate", r.oi, r.si, r.w_new))
for o_uri, s_uri, act, o, s, w_new in edits:
    res = single(o, s, w_new)
    if res is None:
        continue
    dU, lo, hi, dDC, dDist = res
    eff.append({"occ_uri": o_uri, "occupation": olab[o_uri], "skill_uri": s_uri, "action": act,
                "dU": dU, "dU_lo90": lo, "dU_hi90": hi, "dDC_o": dDC, "dDist_o": dDist,
                "noninferior": dU >= m_U, "improves": lo > 0})
EF = pd.DataFrame(eff)

# ============================================================
# 6. REPORT, SAVE, FIGURE
# ============================================================
say()
say(f"Deletable links (not protected): {NP:,}; ZenESCO delete/demote candidates {len(zen)} "
    f"({fz:.1%} of the deletable links); positive control pool {len(ctrl)}")
pv = CV[CV.panel == "delete"].groupby(["strategy", "x"])[["acc", "mrr", "dc", "dist"]].mean().round(3)
say("\nDeletion curves (mean over repetitions)")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    say(pv.unstack(0)["acc"].to_string())
say("\nAt ZenESCO's budget (accuracy | demand coverage | distinctiveness):")
xz = round(fz, 4)
for st in ["random", "frequency pruning", "generic first"]:
    g = CV[(CV.panel == "delete") & (CV.strategy == st)]
    near = g[g.x == g.x.unique()[np.argmin(np.abs(g.x.unique() - xz))]][["acc", "dc", "dist"]].mean()
    say(f"  {st:18s} {near.acc:.3f} | {near.dc:.3f} | {near.dist:.3f}")
zr = CV[(CV.strategy == "ZenESCO") & (CV.x == xz)].iloc[0]
say(f"  {'ZenESCO':18s} {zr.acc:.3f} [{zr.acc_lo:.3f}, {zr.acc_hi:.3f}] | {zr.dc:.3f} | {zr.dist:.3f}")
say(f"  original ESCO      {base['acc']:.3f} | {base['dc']:.3f} | {base['dist']:.3f}")
say("\nAddition curves (accuracy by additions per occupation)")
say(CV[CV.panel == "add"].groupby(["strategy", "x"]).acc.mean().unstack(0).round(3).to_string())
say(f"\nSingle edits: non-inferiority margin m_U = {m_U:.5f} (5th percentile of 300 random single deletions)")
say(EF.groupby("action").agg(n=("dU", "size"), noninferior=("noninferior", "mean"), improves=("improves", "mean"),
                              median_dU=("dU", "median")).round(4).to_string())
say("\nAdditions that clearly improve identification (90% interval above 0), top 15 by dU")
say(EF[(EF.action == "ADD candidate") & EF.improves].nlargest(15, "dU")[["occupation", "skill_uri", "dU", "dDist_o"]]
    .assign(skill=lambda d: d.skill_uri.map(dict(zip(ADD.skill_uri, ADD.skill))))
    [["occupation", "skill", "dU", "dDist_o"]].round(5).to_string(index=False))

OUT.mkdir(parents=True, exist_ok=True)
CV.to_csv(OUT / "curves.csv", index=False)
EF.to_parquet(OUT / "edit_effects.parquet", index=False)
pd.DataFrame([{"m_U": m_U, **base}]).to_csv(OUT / "baseline.csv", index=False)

# ---- Figure F5 (style of the reference perturbation-curve figure) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    STY = {"random": ("#8a8984", "o", "--"), "frequency pruning": ("#eb6834", "v", "-"),
           "generic first": ("#6250d6", "s", "-"), "ZenESCO": ("#e34948", "D", "-"),
           "positive control": ("#1baf7a", "^", ":"), "most frequent (no specificity)": ("#eda100", "P", "-")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3))
    panels = [("delete", "acc", "Identification accuracy", "(a) Deleting links"),
              ("delete", "dc", "Demand coverage", "(b) Deleting links"),
              ("delete", "dist", "Distinctiveness", "(c) Deleting links"),
              ("add", "acc", "Identification accuracy", "(d) Adding links")]
    for ax, (pn, met, ylab, cap) in zip(axs, panels):
        d = CV[CV.panel == pn]
        for st, g in d.groupby("strategy"):
            col, mk, ls = STY[st]
            m = g.groupby("x")[met].mean()
            if g.rep.max() > 0:
                lo, hi = g.groupby("x")[met].quantile(0.025), g.groupby("x")[met].quantile(0.975)
            elif met + "_lo" in g:
                lo, hi = g.groupby("x")[met + "_lo"].mean(), g.groupby("x")[met + "_hi"].mean()
            else:
                lo = hi = None
            xs = m.index.values * (100 if pn == "delete" else 1)
            lw = 1.6 if st == "ZenESCO" else 1.0
            ax.plot(xs, m.values, color=col, marker=mk, ms=3.2, lw=lw, ls=ls, label=st, zorder=4 if st == "ZenESCO" else 3)
            if lo is not None:
                ax.fill_between(xs, lo.values, hi.values, color=col, alpha=0.15, lw=0)
        ax.axhline(base[met], color=INK2, lw=0.5, ls=(0, (1, 2)))
        if pn == "delete":
            zr_ = d[(d.strategy == "ZenESCO") & (d.x == d[d.strategy == "ZenESCO"].x.max())][met].mean()
            ax.annotate("ZenESCO\nbudget", xy=(d[d.strategy == "ZenESCO"].x.max() * 100, zr_), xytext=(8, -14),
                        textcoords="offset points", fontsize=5.8, color="#e34948",
                        arrowprops=dict(arrowstyle="-", color="#e34948", lw=0.5))
        ax.set_xlabel("% of deletable links removed" if pn == "delete" else "links added per occupation")
        ax.set_ylabel(ylab)
        ax.text(0.5, -0.36, cap, transform=ax.transAxes, ha="center", fontsize=8, color=INK)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[3].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh)
            l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.2)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"F5_utility_under_editing.{ext}", dpi=300, bbox_inches="tight")
    log("F5 saved")
except Exception as ex:
    log(f"F5 skipped: {ex!r}")

say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "stage7_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
