"""
ZenESCO - Stage 3: the official ESCO network of the analysed occupations
========================================================================

Objective
    Build the official bipartite graph G = (O u S, E) from ESCO v1.2, describe it for the
    analysed industrial occupations (expectation 1: "what does ESCO look like today?"),
    attach to every official edge the attributes later stages need (type, reuse level,
    digital / transversal / language / green, generality, protection flags), and compute
    the BASELINE values of the structural metrics M1-M7 that Stage 7 will perturb.

    Nothing here uses job ads. Stage 3 is the "before" picture.

Inputs
    BASE\\esco\\  ESCO v1.2 CSV files (names may carry an upload prefix):
        occupations_en.csv, skills_en.csv, occupationSkillRelations_en.csv,
        broaderRelationsSkillPillar_en.csv, skillGroups_en.csv,
        digitalSkillsCollection_en.csv, transversalSkillsCollection_en.csv,
        languageSkillsCollection_en.csv, greenSkillsCollection_en.csv
    BASE\\stage2\\stage2b_targets.csv   (optional) which targets have enough ads.
        If missing, all 65 target occupations are analysed.

Outputs (BASE\\stage3\\)
    official_edges.parquet     one row per official edge of an analysed occupation, with attributes
    occupation_profiles.csv    one row per analysed occupation: profile size, composition,
                               distinctiveness, nearest neighbour, protections
    stage3_report.txt          everything printed to the console
    F2_official_network.png    Figure F2 (profile sizes, skill generality, composition)

Run
    python stage3_official_network.py              # BASE = C:\\skillab_data
    python stage3_official_network.py D:\\other
"""

import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
ESCO_DIR = BASE / "esco"
OUT = BASE / "stage3"

ALPHA = 0.5          # weight of an optional edge relative to an essential one (W_os); sensitivity later
M_ESS = 5            # last-link protection: an occupation keeps at least this many essential links
TAU_SIM = 0.5        # confusability: occupations with cosine similarity >= TAU_SIM
NN_K = 5             # neighbourhood size for NN@k
ROBUST_ORDERS = 200  # random removal orders for M5
SEED = 20261026

# Safety, legal and ethical competences (protection rule 2). Matched on the skill's own
# label and on the labels of all its ancestor groups in the ESCO skills pillar.
# Heuristic first pass; the list is printed for checking and an expert list can extend it.
SAFETY_RE = re.compile(
    r"\b(health and safety|safety|hazard|hazardous|legislation|legal|regulation|regulations|"
    r"regulatory|compliance|data protection|privacy|security|ethic|ethics|ethical)\b", re.I)

# 65 target occupations in 12 clusters (same list as the crawler and Stage 2a)
TARGETS = {1: ('Industrial and manufacturing engineering',
     ['53d2052f-edc0-4f44-8be1-164526412bfe', '6818c837-072a-4120-b913-bd360b7b14d0',
      'd7380260-1386-4e36-98b5-8115ca4a3536', 'b5eaf231-77ad-4a86-8e54-15cc4398aad2',
      'bcc21c63-7eee-4520-8fa7-43eefd389668', '53be4843-8f31-4645-b25c-9a5bbb04f6b6',
      '11833a96-fc27-4b03-a53d-892ac891a2bb']),
 2: ('Automation, instrumentation and control',
     ['bb609566-3ab6-44dd-8f48-cf0b15b96827', '5791a961-4717-4230-a245-a4082a675f04',
      'e6800524-eec2-4c33-af0e-6f36d1d679bc', '2bb8adee-2b84-4d7b-84f9-0620bc09ada8',
      '76abbb82-c103-4d7a-a4c0-14dba4d6199a', '08f5481e-9233-4ef2-9e4b-945224879ce5']),
 3: ('Robotics and mechatronics',
     ['551a6797-3ca9-41fc-9dca-9a982a8fd8a6', '7833d5cd-873d-4fdd-b2f8-9762d68494a7',
      'ab360abd-32e2-4e03-967d-a10758efffa7', 'edf2e989-d7c5-496e-b365-81fc5cb9eb39',
      '5001506f-dd7d-4801-92ac-6f7e93b45133', '869e3a85-6e4c-4e3c-b14f-a8399c1abd2d']),
 4: ('Maintenance and reliability',
     ['615920c5-4f63-4eb3-8b60-afaaed3ab1ff', '8edd8bff-cb59-4c9c-ab0f-59e77d18be48',
      'a9b20949-897b-481f-8642-d1f69263baff', '07dfff6d-a0d0-4202-a697-fe9176c15999']),
 5: ('Quality, testing and metrology',
     ['30c0943c-caa1-411d-b273-26f475971273', '998a0bf4-b1be-42ab-aee5-0096c9be3319',
      'c8fa93eb-7c2c-42c3-b135-c2e825a6615e', 'fdb16700-109e-4d02-949e-3213f278df93',
      'c332af81-4481-4494-9786-aea1f5c5ffcf', 'e6a3ce2e-c0f2-47a7-937a-61ae9abfe1fa']),
 6: ('Smart manufacturing and electrical/electronic engineering',
     ['82616669-cc82-4455-9332-a2df209e9f3e', '418f3a43-a6b2-4101-aaa0-832fe7d82885',
      '4f4d33a1-0b41-4dd9-8a42-aeba5a3974ab', '4d34a9ee-2653-4da0-bef7-f59294577224',
      '77abfaec-a250-4765-95fa-6091e8da1bba', '86ca306c-ab99-420a-9e2a-aa73c5c4de22']),
 7: ('Product design and development engineering',
     ['6fc8f605-98b9-4218-b5e2-91c4c9c55c4d', '579254cf-6d69-4889-9000-9c79dc568644',
      '79fed799-ab3a-43d9-bd91-414c2c3b2f57', '42381741-f3a7-4b26-b35c-da6fad9f1cfd',
      'ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775', 'fde4372d-8293-42de-afbb-8b5a9b274fe7']),
 8: ('Engineering data, CAD and technical documentation',
     ['62979364-6fac-41f2-87ca-202bca19a6ab', 'a4869ae9-2dfe-48cc-bd97-f803880f3407',
      '8cb59bc4-1f39-477e-beea-effed12d186c', '1f958638-066f-470c-9549-a9070ba280a6']),
 9: ('Production management and planning',
     ['6426ada1-4d8c-4612-99e3-91bc569805ee', 'eb9479c6-5fdf-4f45-ace4-28b547b8c7e5',
      'c6bd511a-d966-4df9-a48e-4f800354f268', '40517ed4-2769-4283-a15b-4b0d939bd2d3',
      'b12b14fd-5b74-40c6-afd9-dd35b34e27ed', 'c9862fdd-53b7-4da9-b742-4302d704836c',
      '219d1c30-3849-4de3-b0d6-07da187b6521']),
 10: ('Supply chain planning',
      ['aacc3918-b5d3-484b-9480-5d29aa550d74', 'aeb7c6bd-22c9-43c2-8551-a1cc2560584f',
       'dbdc9f7b-31c6-4f21-a9c5-48253bb69c08', '51cd3a7a-eae6-4dc6-9f24-179d7ec6d230',
       'c8b296be-54ae-48cf-be21-031fa8457937']),
 11: ('Logistics and distribution',
      ['c4071145-a774-4cf4-aefa-ae18a385cca4', '823db06f-25e1-4658-b063-d6ef956b41f5',
       'a2ff6cc3-15fa-4f02-a457-8fd5df7aa29e', '2f5de1ab-eddf-4715-8193-e994912c22ea',
       '632e61c8-4e4b-44c7-b213-6dd5a174b143']),
 12: ('Purchasing and procurement',
      ['377865f0-c327-4599-8c7e-d428c6588edf', '29fa406d-de89-4515-be8d-0fe7bda993f1',
       'c9b2b0a7-0adc-4ca7-bf7c-dc95e18aeac7'])}
OCC_PREFIX = "http://data.europa.eu/esco/occupation/"

T0 = time.time()
REPORT = []


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def esco_file(name, required=True):
    hits = [p for p in ESCO_DIR.glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    if not hits:
        if required:
            sys.exit(f"ESCO file '{name}' not found in {ESCO_DIR}")
        return None
    return hits[0]


def read(name, required=True, **kw):
    p = esco_file(name, required)
    return pd.read_csv(p, dtype=str, keep_default_na=False, **kw) if p is not None else None


# ============================================================
# 1. LOAD ESCO
# ============================================================

log("Loading ESCO v1.2 files")
occ = read("occupations_en.csv")
skl = read("skills_en.csv")
rel = read("occupationSkillRelations_en.csv")
broader = read("broaderRelationsSkillPillar_en.csv")
groups = read("skillGroups_en.csv", required=False)
coll = {}
for tag, fname in [("digital", "digitalSkillsCollection_en.csv"),
                   ("transversal", "transversalSkillsCollection_en.csv"),
                   ("language", "languageSkillsCollection_en.csv"),
                   ("green", "greenSkillsCollection_en.csv")]:
    d = read(fname, required=False)
    coll[tag] = set(d["conceptUri"]) if d is not None else set()

occ_label = dict(zip(occ.conceptUri, occ.preferredLabel))
occ_isco = dict(zip(occ.conceptUri, occ.iscoGroup))
regulated = {u for u, n in zip(occ.conceptUri, occ.regulatedProfessionNote)
             if n and not n.rstrip("/").endswith("unregulated")}
skill_label = dict(zip(skl.conceptUri, skl.preferredLabel))
skill_type = dict(zip(skl.conceptUri, skl.skillType))
skill_reuse = dict(zip(skl.conceptUri, skl.reuseLevel))
# transversal = transversal collection OR reuse level "transversal"
coll["transversal"] |= {u for u, r in skill_reuse.items() if r == "transversal"}

say("ZenESCO - Stage 3 report: the official ESCO network")
say("=" * 70)
say(f"ESCO occupations: {len(occ):,} | skills/knowledge: {len(skl):,} | "
    f"occupation-skill relations: {len(rel):,} "
    f"({(rel.relationType == 'essential').sum():,} essential, {(rel.relationType == 'optional').sum():,} optional)")
say(f"Regulated occupations in ESCO: {len(regulated):,} | collections: "
    + ", ".join(f"{k} {len(v):,}" for k, v in coll.items()))

# ============================================================
# 2. ANALYSED OCCUPATIONS
# ============================================================

target_rows = []
for c, (clabel, keys) in TARGETS.items():
    for k in keys:
        target_rows.append({"occ_uri": OCC_PREFIX + k, "cluster": c, "cluster_label": clabel})
targets = pd.DataFrame(target_rows)
targets["occupation"] = targets.occ_uri.map(occ_label)
missing = targets[targets.occupation.isna()]
if len(missing):
    say(f"WARNING: {len(missing)} target URIs not in ESCO v1.2: {list(missing.occ_uri)}")
targets = targets.dropna(subset=["occupation"])

sel_file = BASE / "stage2" / "stage2b_targets.csv"
if sel_file.exists():
    sel = pd.read_csv(sel_file)
    keep = set(sel.loc[sel["analysed"].astype(bool), "occ_uri"])
    targets["n_ads_H2"] = targets.occ_uri.map(dict(zip(sel.occ_uri, sel.n_H2_2024))).fillna(0).astype(int)
    targets["n_ads_H1"] = targets.occ_uri.map(dict(zip(sel.occ_uri, sel.n_H1_2025))).fillna(0).astype(int)
    analysed = targets[targets.occ_uri.isin(keep)].copy()
    say(f"Analysed occupations: {len(analysed)} of {len(targets)} targets "
        f"(selection from Stage 2b: enough ads in both windows)")
else:
    analysed = targets.copy()
    say(f"Analysed occupations: all {len(analysed)} targets "
        f"(Stage 2b selection not found yet; rerun Stage 3 after Stage 2b)")
A_OCC = list(analysed.occ_uri)
A_SET = set(A_OCC)

# ============================================================
# 3. GLOBAL GRAPH (all ESCO occupations) - needed for generality and IDF
# ============================================================

log("Building the full bipartite graph")
rel = rel[rel.occupationUri.isin(occ_label) & rel.skillUri.isin(skill_label)].copy()
rel["w"] = np.where(rel.relationType == "essential", 1.0, ALPHA)
gen_all = rel.groupby("skillUri").occupationUri.nunique()            # k_s over all ESCO occupations
N_OCC = occ.conceptUri.nunique()

o_idx = {u: i for i, u in enumerate(sorted(rel.occupationUri.unique()))}
s_idx = {u: i for i, u in enumerate(sorted(rel.skillUri.unique()))}
rows = rel.occupationUri.map(o_idx).values
cols = rel.skillUri.map(s_idx).values
W = sparse.csr_matrix((rel.w.values, (rows, cols)), shape=(len(o_idx), len(s_idx)))
k_s = np.asarray((W > 0).sum(axis=0)).ravel()
idf = np.log(N_OCC / np.maximum(k_s, 1))
V = W.multiply(idf.reshape(1, -1)).tocsr()                             # v_os = W_os * log(|O|/k_s)
norms = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
Vn = sparse.diags(1.0 / np.maximum(norms, 1e-12)) @ V

# ============================================================
# 4. SAFETY / LEGAL / ETHICAL SKILLS via the skills pillar hierarchy
# ============================================================

log("Tracing skill ancestors for the safety/legal protection")
parents = defaultdict(set)
for c_, b_ in zip(broader.conceptUri, broader.broaderUri):
    parents[c_].add(b_)
group_label = dict(zip(groups.conceptUri, groups.preferredLabel)) if groups is not None else {}
any_label = {**group_label, **skill_label}


def ancestors(u, max_depth=8):
    seen, q = set(), deque([(u, 0)])
    while q:
        x, d = q.popleft()
        if d >= max_depth:
            continue
        for p in parents.get(x, ()):
            if p not in seen:
                seen.add(p)
                q.append((p, d + 1))
    return seen


def safety_reason(u):
    if SAFETY_RE.search(skill_label.get(u, "")):
        return "label"
    for a in ancestors(u):
        if SAFETY_RE.search(any_label.get(a, "")):
            return "group: " + any_label.get(a, "")
    return ""


# ============================================================
# 5. EDGES OF THE ANALYSED OCCUPATIONS
# ============================================================

E = rel[rel.occupationUri.isin(A_SET)].rename(
    columns={"occupationUri": "occ_uri", "skillUri": "skill_uri", "relationType": "relation"})
E = E[["occ_uri", "skill_uri", "relation"]].copy()
E["occupation"] = E.occ_uri.map(occ_label)
E["cluster"] = E.occ_uri.map(dict(zip(analysed.occ_uri, analysed.cluster)))
E["skill"] = E.skill_uri.map(skill_label)
E["skill_type"] = E.skill_uri.map(skill_type)                         # knowledge / skill/competence
E["reuse_level"] = E.skill_uri.map(skill_reuse)
for tag in ["digital", "transversal", "language", "green"]:
    E["is_" + tag] = E.skill_uri.isin(coll[tag])
E["generality"] = E.skill_uri.map(gen_all).astype(int)                # occupations using the skill (all ESCO)
E["generality_in_set"] = E.groupby("skill_uri").occ_uri.transform("nunique")

log("Applying protection flags")
saf = {u: safety_reason(u) for u in E.skill_uri.unique()}
E["safety_reason"] = E.skill_uri.map(saf)
k_ess = E[E.relation == "essential"].groupby("occ_uri").size()
E["occ_k_ess"] = E.occ_uri.map(k_ess).fillna(0).astype(int)
E["prot_regulated"] = E.occ_uri.isin(regulated)
E["prot_safety"] = E.safety_reason != ""
E["prot_transversal"] = E.is_transversal
E["prot_last_link"] = ((E.relation == "essential") & (E.occ_k_ess <= M_ESS)) | (E.generality == 1)
E["prot_essential"] = E.relation == "essential"     # never deleted automatically, only upgraded/reviewed
E["protected_any"] = E[["prot_regulated", "prot_safety", "prot_transversal", "prot_last_link"]].any(axis=1)

# ============================================================
# 6. BASELINE METRICS
# ============================================================

log("Computing baseline metrics M1-M7")
a_rows = np.array([o_idx[u] for u in A_OCC if u in o_idx])
A_OCC = [u for u in A_OCC if u in o_idx]
S_all = Vn[a_rows] @ Vn.T                                              # analysed x all occupations
S_all = S_all.toarray()
for r, i in enumerate(a_rows):
    S_all[r, i] = -1.0                                                 # exclude self
nn_all = S_all.argmax(axis=1)
inv_o = {i: u for u, i in o_idx.items()}
S_set = S_all[:, a_rows]
nn_set = S_set.argmax(axis=1)

# M3 redundancy of each skill within the analysed subgraph (Latapy et al.)
B = (W[a_rows] > 0).astype(np.int32).tocsc()
used = np.where(np.asarray(B.sum(axis=0)).ravel() > 0)[0]
B = B[:, used]
co = (B @ B.T).toarray()                                               # occupations x occupations shared skills
red = np.full(B.shape[1], np.nan)
for j in range(B.shape[1]):
    occs = B[:, j].nonzero()[0]
    if len(occs) < 2:
        continue
    sub = co[np.ix_(occs, occs)] - 1                                   # shared skills other than s
    iu = np.triu_indices(len(occs), 1)
    red[j] = (sub[iu] > 0).mean()
inv_s = {i: u for u, i in s_idx.items()}
red_map = {inv_s[used[j]]: red[j] for j in range(len(used))}
E["redundancy_in_set"] = E.skill_uri.map(red_map)

# M1 components and non-trivial bridges of the analysed subgraph
import networkx as nx
G = nx.Graph()
G.add_edges_from(("o:" + a, "s:" + b) for a, b in zip(E.occ_uri, E.skill_uri))
comps = list(nx.connected_components(G))
deg = dict(G.degree())
bridges = [(u, v) for u, v in nx.bridges(G)]
nontriv = [(u, v) for u, v in bridges if deg[u] > 1 and deg[v] > 1]

# M5 robustness: AUC of the share of occupations keeping >= M_ESS essential links under random edge removal
rng = np.random.default_rng(SEED)
edge_occ = E.occ_uri.map({u: i for i, u in enumerate(A_OCC)}).values
edge_ess = (E.relation == "essential").values
n_e = len(E)
kess0 = np.bincount(edge_occ[edge_ess], minlength=len(A_OCC))
R_vals = []
for _ in range(ROBUST_ORDERS):
    pos = np.empty(n_e, dtype=np.int64)
    pos[rng.permutation(n_e)] = np.arange(n_e)
    # for each occupation, removal step at which k_ess falls below M_ESS
    fail = np.full(len(A_OCC), 0, dtype=np.int64)
    for o in range(len(A_OCC)):
        if kess0[o] < M_ESS:
            fail[o] = 0
            continue
        p = np.sort(pos[(edge_occ == o) & edge_ess])
        fail[o] = p[kess0[o] - M_ESS] + 1                              # after this many removals it fails
    q = np.arange(1, n_e + 1)
    # F(q) = share of occupations still at >= M_ESS after q removals; R = mean of F over q
    R_vals.append(np.minimum(fail, n_e).sum() / (n_e * len(A_OCC)))
R_G = float(np.mean(R_vals))

# ============================================================
# 7. OCCUPATION PROFILES
# ============================================================

prof = analysed[["occ_uri", "cluster", "cluster_label", "occupation"] +
                [c for c in ["n_ads_H2", "n_ads_H1"] if c in analysed.columns]].copy()
prof["isco"] = prof.occ_uri.map(occ_isco)
prof["regulated"] = prof.occ_uri.isin(regulated)
g = E.groupby("occ_uri")
prof["k"] = prof.occ_uri.map(g.size()).fillna(0).astype(int)
prof["k_ess"] = prof.occ_uri.map(k_ess).fillna(0).astype(int)
prof["k_opt"] = prof.k - prof.k_ess
for col, expr in [("share_knowledge", E.skill_type == "knowledge"),
                  ("share_digital", E.is_digital), ("share_transversal", E.is_transversal),
                  ("share_language", E.is_language), ("share_green", E.is_green),
                  ("share_safety", E.prot_safety), ("share_protected", E.protected_any)]:
    prof[col] = prof.occ_uri.map(expr.astype(float).groupby(E.occ_uri).mean()).astype(float).round(3)
prof["median_generality"] = prof.occ_uri.map(g.generality.median())
prof["share_unique_skills"] = prof.occ_uri.map((E.generality == 1).astype(float).groupby(E.occ_uri).mean()).astype(float).round(3)
pos_of = {u: r for r, u in enumerate(A_OCC)}
prof["nn_esco"] = [occ_label[inv_o[nn_all[pos_of[u]]]] for u in prof.occ_uri]
prof["sim_nn_esco"] = [round(S_all[pos_of[u], nn_all[pos_of[u]]], 3) for u in prof.occ_uri]
prof["dist_esco"] = (1 - prof.sim_nn_esco).round(3)                   # M4 distinctiveness vs all ESCO
prof["nn_in_set"] = [occ_label[A_OCC[nn_set[pos_of[u]]]] for u in prof.occ_uri]
prof["sim_nn_in_set"] = [round(S_set[pos_of[u], nn_set[pos_of[u]]], 3) for u in prof.occ_uri]
prof["dist_in_set"] = (1 - prof.sim_nn_in_set).round(3)
prof["confusable_esco"] = [(S_all[pos_of[u]] >= TAU_SIM).sum() for u in prof.occ_uri]
prof = prof.sort_values(["cluster", "k"], ascending=[True, False])

# ============================================================
# 8. REPORT
# ============================================================

say()
say("1. Size of the official network for the analysed occupations")
say(f"  occupations {len(A_OCC)} | distinct skills {E.skill_uri.nunique():,} | edges {len(E):,} "
    f"({(E.relation == 'essential').sum():,} essential, {(E.relation == 'optional').sum():,} optional)")
say("  profile size k_o: " + ", ".join(f"{q} {v:.0f}" for q, v in
    zip(["min", "p25", "median", "p75", "max"], np.percentile(prof.k, [0, 25, 50, 75, 100]))))
say("  essential links k_ess: " + ", ".join(f"{q} {v:.0f}" for q, v in
    zip(["min", "p25", "median", "p75", "max"], np.percentile(prof.k_ess, [0, 25, 50, 75, 100]))))
say(f"  share essential among edges: {(E.relation == 'essential').mean():.1%}")

say()
say("2. Composition of the edges")
for lab, m in [("knowledge (vs skill/competence)", E.skill_type == "knowledge"),
               ("digital", E.is_digital), ("transversal", E.is_transversal),
               ("language", E.is_language), ("green", E.is_green)]:
    ess = m[E.relation == "essential"].mean()
    opt = m[E.relation == "optional"].mean()
    say(f"  {lab:34s} {m.mean():6.1%}   (essential {ess:6.1%} | optional {opt:6.1%})")
say("  reuse level: " + ", ".join(f"{k} {v:.1%}" for k, v in E.reuse_level.value_counts(normalize=True).items()))

say()
say("3. Skill generality (number of ESCO occupations that use the skill)")
gs = E.drop_duplicates("skill_uri").generality
say("  distinct skills: " + ", ".join(f"{q} {v:.0f}" for q, v in
    zip(["min", "p25", "median", "p75", "p90", "max"], np.percentile(gs, [0, 25, 50, 75, 90, 100]))))
say(f"  skills used by only one ESCO occupation: {(gs == 1).sum():,} ({(gs == 1).mean():.1%})")
say(f"  skills shared by >= 2 analysed occupations: {(E.drop_duplicates('skill_uri').generality_in_set >= 2).sum():,}")
say("  most general skills in these profiles:")
top_g = E.drop_duplicates("skill_uri").nlargest(10, "generality")[["skill", "generality", "generality_in_set"]]
say(top_g.to_string(index=False))

say()
say("4. Protection flags (protection rules; an edge can have several)")
for lab, col in [("regulated occupation", "prot_regulated"), ("safety / legal / ethical skill", "prot_safety"),
                 ("transversal skill", "prot_transversal"), ("last link", "prot_last_link"),
                 ("any of the four", "protected_any"), ("essential (no automatic deletion)", "prot_essential")]:
    say(f"  {lab:36s} {E[col].sum():6,} edges ({E[col].mean():.1%})")
opt_unprot = ((E.relation == "optional") & ~E.protected_any).sum()
say(f"  -> optional, unprotected edges (eligible pool for deletion): {opt_unprot:,} ({opt_unprot / len(E):.1%})")
say(f"  -> essential, unprotected edges (eligible for upgrade to optional): "
    f"{((E.relation == 'essential') & ~E.protected_any).sum():,}")
say(f"  regulated analysed occupations: {list(prof.loc[prof.regulated, 'occupation'])}")
say("  safety/legal skills found, sample for checking:")
s_s = E[E.prot_safety].drop_duplicates("skill_uri")[["skill", "safety_reason"]]
say(s_s.sample(min(15, len(s_s)), random_state=1).to_string(index=False))

say()
say("5. Baseline structural metrics (Stage 7 measures their change)")
say(f"  M1 connected components {len(comps)} | giant component {max(len(c) for c in comps):,} nodes "
    f"| bridges {len(bridges):,} of which non-trivial {len(nontriv)}")
say(f"  M2 coverage: median k_o {prof.k.median():.0f}, median k_ess {prof.k_ess.median():.0f}; "
    f"occupations with k_ess < {M_ESS}: {(prof.k_ess < M_ESS).sum()}")
rr = E.drop_duplicates("skill_uri").redundancy_in_set.dropna()
say(f"  M3 skill redundancy (skills in >= 2 analysed occupations, n={len(rr):,}): "
    f"median {rr.median():.2f}, share fully redundant (=1) {(rr == 1).mean():.1%}")
say(f"  M4 distinctiveness vs all ESCO: median {prof.dist_esco.median():.3f}, "
    f"min {prof.dist_esco.min():.3f}; within analysed set: median {prof.dist_in_set.median():.3f}")
say(f"     occupations with a near-twin in ESCO (sim >= {TAU_SIM}): {(prof.confusable_esco > 0).sum()}")
say(f"  M5 robustness R(G) (share keeping >= {M_ESS} essential links, {ROBUST_ORDERS} random orders): {R_G:.3f}")
say(f"  M7 similarity matrix: {len(A_OCC)} x {N_OCC} (baseline for NN@{NN_K})")
say("  M6 centrality and M8 alignment are computed in Stage 7 (M8 needs ads).")

say()
say("6. Occupation profiles")
cols_show = ["cluster", "occupation", "k", "k_ess", "k_opt", "share_digital", "share_transversal",
             "share_protected", "dist_esco", "nn_esco", "sim_nn_esco"]
with pd.option_context("display.width", 250, "display.max_colwidth", 40):
    say(prof[cols_show].to_string(index=False))

say()
say("7. Least distinctive occupations (closest ESCO neighbour)")
say(prof.nsmallest(10, "dist_esco")[["occupation", "nn_esco", "sim_nn_esco"]].to_string(index=False))

# ============================================================
# 9. SAVE AND FIGURE F2
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)
E.to_parquet(OUT / "official_edges.parquet", index=False)
prof.to_csv(OUT / "occupation_profiles.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_ESS, C_OPT, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(11, max(6, 0.14 * len(prof) + 1.5)))
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.35)
    # (a) profile sizes
    ax = fig.add_subplot(gsp[:, 0])
    p = prof.sort_values("k")
    y = np.arange(len(p))
    ax.barh(y, p.k_ess, color=C_ESS, height=0.7, label="essential", edgecolor="white", linewidth=0.5)
    ax.barh(y, p.k_opt, left=p.k_ess, color=C_OPT, height=0.7, label="optional", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(p.occupation, fontsize=6)
    ax.set_xlabel("ESCO skills in the profile")
    ax.set_title("(a) Profile size per occupation", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    # (b) generality
    ax = fig.add_subplot(gsp[0, 1])
    bins = np.logspace(0, np.log10(E.generality.max() + 1), 25)
    ax.hist([E.loc[E.relation == "essential", "generality"], E.loc[E.relation == "optional", "generality"]],
            bins=bins, stacked=True, color=[C_ESS, C_OPT], label=["essential edges", "optional edges"],
            edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("skill generality: ESCO occupations using the skill (log)")
    ax.set_ylabel("edges")
    ax.set_title("(b) How generic the linked skills are", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False)
    # (c) composition
    ax = fig.add_subplot(gsp[1, 1])
    cats = [("knowledge", E.skill_type == "knowledge"), ("digital", E.is_digital),
            ("transversal", E.is_transversal), ("language", E.is_language), ("green", E.is_green),
            ("safety/legal", E.prot_safety)]
    x = np.arange(len(cats))
    ess_v = [m[E.relation == "essential"].mean() * 100 for _, m in cats]
    opt_v = [m[E.relation == "optional"].mean() * 100 for _, m in cats]
    ax.bar(x - 0.2, ess_v, 0.38, color=C_ESS, label="essential")
    ax.bar(x + 0.2, opt_v, 0.38, color=C_OPT, label="optional")
    for xi, v in zip(x, ess_v):
        ax.text(xi - 0.2, v + 0.8, f"{v:.0f}", ha="center", fontsize=6, color=MUTED)
    for xi, v in zip(x, opt_v):
        ax.text(xi + 0.2, v + 0.8, f"{v:.0f}", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in cats], fontsize=7)
    ax.set_ylabel("% of edges")
    ax.set_title("(c) What kind of skills the edges point to", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False)
    fig.savefig(OUT / "F2_official_network.png", dpi=200, bbox_inches="tight")
    log("Figure F2 saved")
except Exception as ex:   # the figure is optional; the tables are what matters
    log(f"Figure skipped: {ex}")

say()
say(f"Runtime: {time.time() - T0:.0f}s")
(OUT / "stage3_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"Done. Outputs in {OUT}")
