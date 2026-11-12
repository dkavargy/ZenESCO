"""
ZenESCO - Stage 8: sensitivity analysis of the main settings
============================================================

Question
    Do the conclusions of the paper depend on the particular values chosen for the main settings?
    Conclusions checked in every scenario (held-out window H1 2025, scored exactly as in Stage 9 / Table 4):
      C1  ZenESCO raises identification accuracy over the original profiles (95% bootstrap interval of the gain > 0)
      C2  ZenESCO's gain is larger than the gain of frequency pruning
      C3  ZenESCO raises the coverage of the top-K demanded skills
      C4  no essential link is deleted outright (essential links can only be demoted)
    and how much the set of recommended changes overlaps with the paper's run (Jaccard index).

Settings varied (one at a time around the paper's values)
    CRED            credibility level of every evidence rule (Stages 4, 5-6, 9)           0.90 | 0.95 | 0.99
    ALPHA_POW       significance level of the power test (Eq. 4; Stages 4, 5-6)        0.01 | 0.05 | 0.10
    PRIOR_STRENGTH  Beta prior worth kappa pseudo-ads (Stages 4, 9)                       1 | 2 | 5
    ALPHA           weight of an optional link (Eq. 1; Stages 7, 9 and the scoring)     0.3 | 0.5 | 0.7
    STAB            stability thresholds (accept, Firm) of the bootstrap (Stage 9)     (0.7, 0.85) | (0.8, 0.9) | (0.9, 0.95)
    TOP_K           size of the demand set for coverage (evaluation only)                 10 | 20 | 30
How
    For each scenario the pipeline is re-run from the first stage that uses the setting, on a scenario copy of
    BASE: the scenario folder links (junction/symlink, or copies) the unchanged inputs and earlier stage outputs,
    and the stage scripts in THIS folder are copied with the one constant line replaced. Your own scripts and
    outputs are never modified. The expert review (stage10) is applied wherever its items still occur; items that
    only arise under a scenario stay "Expert review" (not changed).
    The paper's run (BASE\\stage6, stage7, stage9) is the baseline and is not re-run.

Inputs   BASE\\stage1, stage2, stage3, stage4, stage6, stage7, stage9, stage10, esco (existing outputs)
         stage4_evidence.py, stage5_6_screening.py, stage7_perturbation.py, stage9_final_adjustment.py (this folder)
Outputs  BASE\\stage8\\scenarios\\<name>\\...         full stage outputs per scenario (logs in _logs)
         BASE\\stage8\\sensitivity.csv, sensitivity_report.txt, F10_sensitivity.png/pdf
Run      python stage8_sensitivity.py                       # all scenarios; finished scenarios are skipped
         python stage8_sensitivity.py C:\\skillab_data cred_0.90 stab_0.7   # only the named scenarios
         python stage8_sensitivity.py C:\\skillab_data --evaluate-only      # only re-evaluate finished scenarios
Re-run   delete the scenario folder in Windows Explorer (deleting a folder link never deletes the linked original)
Runtime  a scenario that starts at Stage 4 takes about as long as Stages 4 + 5-6 + 7 + 9 together.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = Path(ARGS.pop(0)) if ARGS and Path(ARGS[0]).is_dir() else Path(r"C:\skillab_data")
ONLY = ARGS                                   # optional scenario names
EVAL_ONLY = "--evaluate-only" in sys.argv
CODE = Path(__file__).resolve().parent
OUT = BASE / "stage8"
SCN = OUT / "scenarios"
EVAL_WINDOW = "H1_2025"
BOOT = 1000
SEED = 20261026
BASE_VALUES = {"CRED": 0.95, "ALPHA_POW": 0.05, "PRIOR_STRENGTH": 2.0, "ALPHA": 0.5, "STAB": (0.8, 0.9), "TOP_K": 20}

# where each setting lives: (script, regex of the whole constant line, replacement template)
STAGES = ["stage4_evidence.py", "stage5_6_screening.py", "stage7_perturbation.py", "stage9_final_adjustment.py"]
OUTDIR = {"stage4_evidence.py": "stage4", "stage5_6_screening.py": "stage6",
          "stage7_perturbation.py": "stage7", "stage9_final_adjustment.py": "stage9"}
WHERE = {
    "CRED": [("stage4_evidence.py", r"^CRED = .*$", "CRED = {v}"),
             ("stage5_6_screening.py", r"^CRED = .*$", "CRED = {v}"),
             ("stage9_final_adjustment.py", r"^CRED = .*$", "CRED = {v}")],
    "ALPHA_POW": [("stage4_evidence.py", r"^ALPHA_POW = .*$", "ALPHA_POW = {v}"),
                  ("stage5_6_screening.py", r"^ALPHA_POW = .*$", "ALPHA_POW = {v}")],
    "PRIOR_STRENGTH": [("stage4_evidence.py", r"^PRIOR_STRENGTH = .*$", "PRIOR_STRENGTH = {v}"),
                       ("stage9_final_adjustment.py", r"^PRIOR_STRENGTH = .*$", "PRIOR_STRENGTH = {v}")],
    "ALPHA": [("stage7_perturbation.py", r"^ALPHA = .*$", "ALPHA = {v}"),
              ("stage9_final_adjustment.py", r"^ALPHA = .*$", "ALPHA = {v}")],
    "STAB": [("stage9_final_adjustment.py", r"^STAB_OK, STAB_FIRM, STAB_LOW = .*$",
              "STAB_OK, STAB_FIRM, STAB_LOW = {v[0]}, {v[1]}, 0.5")],
    "TOP_K": [],                                   # evaluation only
}
SCENARIOS = [  # name, setting, value
    ("cred_0.90", "CRED", 0.90), ("cred_0.99", "CRED", 0.99),
    ("power_0.01", "ALPHA_POW", 0.01), ("power_0.10", "ALPHA_POW", 0.10),
    ("prior_1", "PRIOR_STRENGTH", 1.0), ("prior_5", "PRIOR_STRENGTH", 5.0),
    ("alpha_0.3", "ALPHA", 0.3), ("alpha_0.7", "ALPHA", 0.7),
    ("stab_0.7", "STAB", (0.7, 0.85)), ("stab_0.9", "STAB", (0.9, 0.95)),
    ("topk_10", "TOP_K", 10), ("topk_30", "TOP_K", 30),
]
LABEL = {"CRED": "credibility level", "ALPHA_POW": "power-test level", "PRIOR_STRENGTH": "prior strength \u03ba",
         "ALPHA": "optional weight \u03b1", "STAB": "stability thresholds", "TOP_K": "coverage set K"}
T0 = time.time()
REPORT = []


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def link(src, dst):
    """directory link that works on Windows without admin rights (junction), else symlink, else copy"""
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, text=True)
        if r.returncode == 0:
            return
    log(f"  copying {src.name} (links not available)")
    shutil.copytree(str(src), str(dst))


def patched(script, setting, value):
    txt = (CODE / script).read_text(encoding="utf-8")
    for sc, pat, rep in WHERE.get(setting, []):
        if sc != script:
            continue
        new, k = re.subn(pat, rep.format(v=value) + "   # <- sensitivity scenario (Stage 8)", txt, count=1, flags=re.M)
        if k != 1:
            sys.exit(f"Could not find the line for {setting} in {script} (pattern {pat}). Is it the current version?")
        txt = new
    return txt


def run_scenario(name, setting, value):
    sdir = SCN / name
    done = sdir / "stage9" / "R1_all_links.csv"
    if setting == "TOP_K":
        return BASE                                   # evaluation-only setting: the paper's run
    if done.exists():
        log(f"{name}: already finished, skipped (delete {sdir} to re-run)")
        return sdir
    if EVAL_ONLY:
        log(f"{name}: not run yet (--evaluate-only)")
        return None
    first = min(STAGES.index(sc) for sc, _, _ in WHERE[setting])
    to_run = STAGES[first:]
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "_code").mkdir(exist_ok=True)
    (sdir / "_logs").mkdir(exist_ok=True)
    for d in ["stage1", "stage2", "stage3", "stage10", "esco"]:
        if (BASE / d).exists():
            link(BASE / d, sdir / d)
    for sc in STAGES[:first]:                         # earlier stages: the paper's outputs
        link(BASE / OUTDIR[sc], sdir / OUTDIR[sc])
    for sc in to_run:
        od = sdir / OUTDIR[sc]
        if od.is_symlink():
            od.unlink()
        f = sdir / "_code" / sc
        f.write_text(patched(sc, setting, value), encoding="utf-8")
        log(f"{name}: running {sc}")
        with open(sdir / "_logs" / (sc + ".log"), "w", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, str(f), str(sdir)], cwd=str(sdir / "_code"), stdout=lf,
                               stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            log(f"{name}: {sc} FAILED - see {sdir / '_logs' / (sc + '.log')}")
            return None
    return sdir


# ============================================================
# 1. RUN SCENARIOS
# ============================================================
todo = [s for s in SCENARIOS if not ONLY or s[0] in ONLY]
results = {"baseline": (BASE, "baseline", None)}
for name, setting, value in todo:
    d = run_scenario(name, setting, value)
    if d is not None:
        results[name] = (d, setting, value)

# ============================================================
# 2. EVALUATION (same scoring as Stage 9)
# ============================================================
log("Evaluating")


def esco_file(name):
    hits = [p for p in (BASE / "esco").glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    return hits[0] if hits else None


EV0 = pd.read_parquet(BASE / "stage6" / "screened_links.parquet")
OCC = sorted(EV0.occ_uri.unique())
oi = {o: i for i, o in enumerate(OCC)}
ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet", columns=["ad_id", "window", "analysable"])
ads = ads[ads.analysable.astype(bool) & (ads.window == EVAL_WINDOW)].copy()
ads["ad_id"] = ads.ad_id.astype(str)
occ_of = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ_of["ad_id"] = occ_of.ad_id.astype(str)
test = ads.merge(occ_of, on="ad_id")
test = test[test.occ_uri.isin(set(OCC))].reset_index(drop=True)
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet", columns=["ad_id", "skill_uri"])
ask["ad_id"] = ask.ad_id.astype(str)
at = ask[ask.ad_id.isin(set(test.ad_id))].drop_duplicates()
rel = pd.read_csv(esco_file("occupationSkillRelations_en.csv"), dtype=str, usecols=["occupationUri", "skillUri"])
N_ESCO_OCC = rel.occupationUri.nunique()
gen = rel.groupby("skillUri").occupationUri.nunique()
lang = set()
if esco_file("languageSkillsCollection_en.csv") is not None:
    lang = set(pd.read_csv(esco_file("languageSkillsCollection_en.csv"), dtype=str).conceptUri)

CHANGE = ["Delete", "Upgrade -> optional", "Upgrade -> essential", "Add (essential)", "Add (optional)"]
PROFS = {}
for name, (d, setting, value) in results.items():
    R1 = pd.read_csv(d / "stage9" / "R1_all_links.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    PROFS[name] = R1
SK = sorted(set(at.skill_uri) | set(EV0.skill_uri) | {s for R1 in PROFS.values() for s in R1.skill_uri})
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
rng = np.random.default_rng(SEED)
BIDX = rng.integers(0, len(test), size=(BOOT, len(test)))


def profile(R1, alpha, which="zenesco"):
    P = {o: {} for o in OCC}
    for o, s, r in zip(EV0.occ_uri, EV0.skill_uri, EV0.relation):
        P[o][s] = 1.0 if r == "essential" else alpha
    if which == "original":
        return P
    if which == "frequency pruning":
        for o, s, n in zip(EV0.occ_uri, EV0.skill_uri, EV0.n):
            if n == 0:
                P[o].pop(s, None)
        return P
    for o, s, rec in zip(R1.occ_uri, R1.skill_uri, R1.recommendation):
        if o not in P:
            continue
        if rec == "Delete":
            P[o].pop(s, None)
        elif rec in ("Upgrade -> optional", "Add (optional)"):
            P[o][s] = alpha
        elif rec in ("Upgrade -> essential", "Add (essential)"):
            P[o][s] = 1.0
    return P


def score(P, K):
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
    cov = []
    for o in OCC:
        h = h1prev[(h1prev.occ_uri == o) & ~h1prev.skill_uri.isin(lang)].nlargest(K, "p")
        if len(h):
            cov.append(h.skill_uri.isin(set(P[o])).mean())
    return acc, rr, float(np.mean(cov)), float(np.median(1 - C.max(axis=1)))


def changes(R1):
    c = R1[R1.recommendation.isin(CHANGE)]
    return set(zip(c.occ_uri, c.skill_uri, c.recommendation))


base_changes = changes(PROFS["baseline"])
rows = []
order = [("baseline", "baseline", None)] + [(n, s, v) for n, s, v in todo if n in results]
for name, setting, value in order:
    R1 = PROFS[name]
    alpha = value if setting == "ALPHA" else BASE_VALUES["ALPHA"]
    K = value if setting == "TOP_K" else BASE_VALUES["TOP_K"]
    acc0, rr0, cov0, dist0 = score(profile(R1, alpha, "original"), K)
    accf, _, covf, distf = score(profile(R1, alpha, "frequency pruning"), K)
    acc, rr, cov, dist = score(profile(R1, alpha), K)
    gain = acc[BIDX].mean(axis=1) - acc0[BIDX].mean(axis=1)
    gainf = accf[BIDX].mean(axis=1) - acc0[BIDX].mean(axis=1)
    diff = acc[BIDX].mean(axis=1) - accf[BIDX].mean(axis=1)
    ch = changes(R1)
    rec = R1.recommendation.value_counts()
    ess_del = int(((R1.recommendation == "Delete") & (R1.esco_relation == "essential")).sum())
    conf = R1[R1.recommendation.isin(CHANGE)].confidence.value_counts()
    rows.append({
        "scenario": name, "setting": setting, "label": LABEL.get(setting, "paper's run"),
        "value": "" if value is None else (f"{value[0]}/{value[1]}" if isinstance(value, tuple) else value),
        "acc_original": acc0.mean(), "acc_freq_pruning": accf.mean(), "acc_zenesco": acc.mean(),
        "gain_lo": np.percentile(gain, 2.5), "gain_hi": np.percentile(gain, 97.5),
        "gain_vs_freq_lo": np.percentile(diff, 2.5), "gain_vs_freq_hi": np.percentile(diff, 97.5),
        "mrr_original": rr0.mean(), "mrr_zenesco": rr.mean(),
        "cov_original": cov0, "cov_zenesco": cov, "dist_original": dist0, "dist_zenesco": dist,
        "n_delete": int(rec.get("Delete", 0)), "n_demote": int(rec.get("Upgrade -> optional", 0)),
        "n_upgrade": int(rec.get("Upgrade -> essential", 0)),
        "n_add": int(rec.get("Add (essential)", 0) + rec.get("Add (optional)", 0)),
        "n_expert_open": int(rec.get("Expert review", 0)),
        "n_firm": int(conf.get("Firm", 0)), "n_tentative": int(conf.get("Tentative", 0)),
        "essential_deleted": ess_del,
        "jaccard_with_paper": len(ch & base_changes) / max(len(ch | base_changes), 1),
        "share_of_paper_changes_kept": len(ch & base_changes) / max(len(base_changes), 1),
        "C1_gain_positive": bool(np.percentile(gain, 2.5) > 0),
        "C2_beats_freq_pruning": bool(np.percentile(diff, 2.5) > 0),
        "C3_coverage_up": bool(cov > cov0), "C4_no_essential_deleted": ess_del == 0,
    })
RES = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
RES.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")

say("ZenESCO - Stage 8 report: sensitivity of the conclusions to the main settings")
say("=" * 78)
say(f"Held-out window {EVAL_WINDOW}: {len(test):,} ads of {len(OCC)} occupations; {BOOT} bootstrap resamples. "
    f"Paper values: " + ", ".join(f"{k} = {v}" for k, v in BASE_VALUES.items()))
hdr = (f"{'scenario':12s} {'setting':22s} {'value':>9s} {'acc0':>6s} {'accF':>6s} {'accZ':>6s} {'gain 95% CI':>17s} "
       f"{'cov0':>5s} {'covZ':>5s} {'distZ':>5s} {'del':>4s} {'dem':>4s} {'upg':>4s} {'add':>4s} {'open':>4s} "
       f"{'Jacc':>5s} {'C1-C4':>6s}")
say(hdr)
say("-" * len(hdr))
for r in RES.to_dict("records"):
    cs = "".join("Y" if r[c] else "n" for c in ["C1_gain_positive", "C2_beats_freq_pruning", "C3_coverage_up",
                                                "C4_no_essential_deleted"])
    say(f"{r['scenario']:12s} {r['label']:22s} {str(r['value']):>9s} {r['acc_original']:6.3f} {r['acc_freq_pruning']:6.3f} "
        f"{r['acc_zenesco']:6.3f} [{r['gain_lo']:+.3f},{r['gain_hi']:+.3f}] {r['cov_original']:5.2f} "
        f"{r['cov_zenesco']:5.2f} {r['dist_zenesco']:5.2f} {r['n_delete']:4d} {r['n_demote']:4d} {r['n_upgrade']:4d} "
        f"{r['n_add']:4d} {r['n_expert_open']:4d} {r['jaccard_with_paper']:5.2f} {cs:>6s}")
allc = RES[["C1_gain_positive", "C2_beats_freq_pruning", "C3_coverage_up", "C4_no_essential_deleted"]].all(axis=1)
say("")
say(f"Conclusions C1-C4 hold in {int(allc.sum())} of {len(RES)} runs "
    f"(C1 gain > 0: {int(RES.C1_gain_positive.sum())}, C2 beats frequency pruning: {int(RES.C2_beats_freq_pruning.sum())}, "
    f"C3 coverage up: {int(RES.C3_coverage_up.sum())}, C4 no essential deletion: {int(RES.C4_no_essential_deleted.sum())}).")
sc_ = RES[RES.scenario != "baseline"]
if len(sc_):
    say(f"Across scenarios: accuracy {sc_.acc_zenesco.min():.3f}-{sc_.acc_zenesco.max():.3f} (paper {RES.acc_zenesco.iloc[0]:.3f}); "
        f"changes overlap with the paper's run: Jaccard {sc_.jaccard_with_paper.min():.2f}-{sc_.jaccard_with_paper.max():.2f}, "
        f"{sc_.share_of_paper_changes_kept.min():.0%}-{sc_.share_of_paper_changes_kept.max():.0%} of its changes kept.")
say("Notes: Jaccard compares (occupation, skill, recommendation) of all changes with the paper's run. 'open' = items left "
    "for expert review because they arise only under the scenario (the review of the paper's run is applied where the "
    "items coincide). Scenarios share the H1 2025 caveat of Stage 9 (held-out tests used in the selection).")

# ============================================================
# 3. FIGURE F10 (colours of Fig. 6 / Fig. 9)
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    RED, GREY, ORANGE, BLUE, GREEN, VIOLET = "#e34948", "#c9c8c2", "#eb6834", "#2a78d6", "#1baf7a", "#6250d6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
    R = RES.iloc[::-1].reset_index(drop=True)
    ylab = ["paper's run" if r.scenario == "baseline" else f"{r.label} = {r.value}" for r in R.itertuples()]
    y = np.arange(len(R))
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 0.24 * len(R) + 1.3), sharey=True,
                            gridspec_kw={"width_ratios": [1.3, 1, 1.3, 0.8]})
    ax = axs[0]
    ax.scatter(R.acc_original, y, color=GREY, s=16, label="original ESCO", zorder=3)
    ax.scatter(R.acc_freq_pruning, y, color=ORANGE, marker="v", s=16, label="frequency pruning", zorder=3)
    lo = R.acc_original + R.gain_lo
    hi = R.acc_original + R.gain_hi
    ax.hlines(y, lo, hi, color=RED, lw=1.6, alpha=0.5, zorder=2)
    ax.scatter(R.acc_zenesco, y, color=RED, marker="D", s=16, label="ZenESCO", zorder=4)
    ax.set_xlabel("Identification accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.6 / len(R)), ncol=1, fontsize=6)
    ax = axs[1]
    ax.scatter(R.cov_original, y, color=GREY, s=16, zorder=3)
    ax.scatter(R.cov_zenesco, y, color=RED, marker="D", s=16, zorder=4)
    ax.hlines(y, R.cov_original, R.cov_zenesco, color=RED, lw=0.8, alpha=0.4)
    ax.set_xlabel("Coverage of top-K skills")
    ax = axs[2]
    left = np.zeros(len(R))
    for col, c, lab in [("n_delete", RED, "delete"), ("n_demote", ORANGE, "demote"), ("n_upgrade", BLUE, "upgrade"),
                        ("n_add", GREEN, "add"), ("n_expert_open", GREY, "open review")]:
        ax.barh(y, R[col], left=left, color=c, height=0.62, label=lab, zorder=3)
        left = left + R[col].values
    ax.set_xlabel("Recommended changes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.6 / len(R)), ncol=3, fontsize=6, handlelength=1)
    ax = axs[3]
    ax.barh(y, R.jaccard_with_paper, color=VIOLET, height=0.62, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Overlap with paper (Jaccard)")
    for a_ in axs:
        a_.grid(axis="x", color=GRID, lw=0.5)
        a_.set_axisbelow(True)
        a_.axhline(len(R) - 1 - 0.5, color=MUTED, lw=0.4, ls=(0, (1, 2)))
    for a_, cap in zip(axs, ["(a) Identification", "(b) Demand coverage", "(c) Changes", "(d) Stability of the set"]):
        a_.text(0.5, -0.9 / (0.24 * len(R) + 1.3), cap, transform=a_.transAxes, ha="center", va="top", fontsize=8, color=INK)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(ylab, fontsize=6.5)
    for t in axs[0].get_yticklabels():
        if t.get_text() == "paper's run":
            t.set_fontweight("bold")
    fig.tight_layout(w_pad=0.8)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"F10_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    log("F10 saved")
except Exception as ex:
    import traceback
    traceback.print_exc()
    log(f"F10 skipped: {ex!r}")

say(f"\nRuntime: {time.time() - T0:.0f}s")
(OUT / "sensitivity_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
