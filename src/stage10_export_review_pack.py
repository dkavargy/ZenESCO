"""
ZenESCO - Stage 10 (step 1): export the expert-review pack
==========================================================

Makes ONE file, review_pack.csv, with every item the reviewer must judge:
  - all items with recommendation "Expert review" (conflicting evidence)
  - an audit sample of the automatic changes: up to AUDIT_PER_TYPE per change type
    (Delete, Upgrade -> optional, Upgrade -> essential, Add), drawn at random (fixed seed)
For each item: occupation, skill, ESCO definitions, current ESCO status, ZenESCO recommendation and
evidence, and up to N_TITLES titles of real H2 2024 ads of that occupation that mention the skill
(so extraction errors and boilerplate can be recognised).

Phase A columns (relevance judged WITHOUT evidence) come first in the file; the evidence columns follow.

ROUND 2 (automatic): if BASE\\stage10\\review_results.xlsx (or .csv) already exists, the script instead exports
every item of the current R1 that is a change or an open Expert review and has NOT been reviewed yet, so that every
change in the final taxonomy is checked by the expert. Output: review_pack_2.csv, item ids continue after the last one.

Run   python stage10_export_review_pack.py            # BASE = C:\\skillab_data
Out   BASE\\stage10\\review_pack.csv  or  review_pack_2.csv   (send this file)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
OUT = BASE / "stage10"
AUDIT_PER_TYPE = 15
N_TITLES = 5
SEED = 20261026


def esco_file(name):
    hits = [p for p in (BASE / "esco").glob("*.csv") if p.name == name or p.name.endswith("_" + name)]
    return hits[0] if hits else None


R1 = pd.read_csv(BASE / "stage9" / "R1_all_links.csv")
CHANGE = ["Delete", "Upgrade -> optional", "Upgrade -> essential", "Add (essential)", "Add (optional)"]


def read_review(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=None, engine="python")
    try:
        return pd.read_excel(path, sheet_name="Review", dtype=str, engine="openpyxl").fillna("")
    except Exception:
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Review"]
        rows = [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
        return pd.DataFrame(rows[1:], columns=rows[0])


done = [p for p in [OUT / "review_results.xlsx", OUT / "review_results.csv"] if p.exists()]
ROUND2 = bool(done)
if ROUND2:
    rv = read_review(done[0])
    seen = set(zip(rv.occ_uri, rv.skill_uri))
    last = max(int(i[1:]) for i in rv.item_id if str(i).startswith("R"))
    todo = R1[(R1.recommendation.isin(CHANGE + ["Expert review"]))
              & np.array([(o, s_) not in seen for o, s_ in zip(R1.occ_uri, R1.skill_uri)])].copy()
    pack = todo.assign(review_type=np.where(todo.recommendation == "Expert review", "expert review (conflict)",
                                            "check of automatic change")).reset_index(drop=True)
    print(f"Round 2: {len(seen)} items already reviewed; {len(pack)} changes / open items not yet reviewed")
else:
    rev = R1[R1.recommendation == "Expert review"].assign(review_type="expert review (conflict)")
    auto = R1[R1.recommendation.isin(CHANGE)].copy()
    auto["kind"] = auto.recommendation.str.replace(r" \(.*\)", "", regex=True)
    aud = pd.concat([g.sample(min(AUDIT_PER_TYPE, len(g)), random_state=SEED) for _, g in auto.groupby("kind")])
    aud = aud.drop(columns="kind").assign(review_type="audit of automatic change")
    pack = pd.concat([rev, aud], ignore_index=True)
    last = 0

sk = pd.read_csv(esco_file("skills_en.csv"), dtype=str, keep_default_na=False,
                 usecols=["conceptUri", "description", "skillType", "reuseLevel"])
oc = pd.read_csv(esco_file("occupations_en.csv"), dtype=str, keep_default_na=False,
                 usecols=["conceptUri", "description"])
pack["skill_definition"] = pack.skill_uri.map(dict(zip(sk.conceptUri, sk.description))).str.slice(0, 400)
pack["skill_type"] = pack.skill_uri.map(dict(zip(sk.conceptUri, sk.skillType)))
pack["reuse_level"] = pack.skill_uri.map(dict(zip(sk.conceptUri, sk.reuseLevel)))
pack["occupation_definition"] = pack.occ_uri.map(dict(zip(oc.conceptUri, oc.description))).str.slice(0, 400)

# example ad titles (H2 2024 ads of the occupation that mention the skill)
ads = pd.read_parquet(BASE / "stage1" / "ads_clean.parquet", columns=["ad_id", "window", "title_raw", "analysable"])
ads = ads[ads.analysable & (ads.window == "H2_2024")]
ads["ad_id"] = ads.ad_id.astype(str)
occ = pd.read_parquet(BASE / "stage2" / "ad_occupation.parquet", columns=["ad_id", "occ_uri"])
occ["ad_id"] = occ.ad_id.astype(str)
ads = ads.merge(occ, on="ad_id")
ask = pd.read_parquet(BASE / "stage1" / "ad_skills_clean.parquet")
ask["ad_id"] = ask.ad_id.astype(str)
need = set(zip(pack.occ_uri, pack.skill_uri))
m = ask[ask.skill_uri.isin(set(pack.skill_uri))].merge(ads[["ad_id", "occ_uri", "title_raw"]], on="ad_id")
m = m[[(o, s) in need for o, s in zip(m.occ_uri, m.skill_uri)]]
ex = {}
for (o, s_), t in m.groupby(["occ_uri", "skill_uri"]).title_raw:
    t = t.drop_duplicates()
    ex[(o, s_)] = " || ".join(t.sample(min(N_TITLES, len(t)), random_state=SEED))
pack["example_ad_titles_H2"] = [ex.get((o, s_), "") for o, s_ in zip(pack.occ_uri, pack.skill_uri)]

pack = pack.sample(frac=1, random_state=SEED).reset_index(drop=True)          # shuffle: no ordering by evidence
pack.insert(0, "item_id", [f"R{last + i + 1:04d}" for i in range(len(pack))])
phase_a = ["item_id", "occupation", "occupation_definition", "skill", "skill_definition", "skill_type",
           "reuse_level", "esco_relation", "in_esco"]
evidence = ["review_type", "recommendation", "confidence", "reason", "share_ads_H2", "share_ads_H1", "lift",
            "pr_specific", "stability", "dMRR_heldout", "dDistinct_o", "generality", "semantic_pct",
            "structural_pct", "example_ad_titles_H2", "occ_uri", "skill_uri"]
pack = pack[[c for c in phase_a + evidence if c in pack.columns]]
OUT.mkdir(parents=True, exist_ok=True)
name = "review_pack_2.csv" if ROUND2 else "review_pack.csv"
pack.to_csv(OUT / name, index=False, encoding="utf-8-sig")
print(f"{name}: {len(pack)} items -> {OUT}  (send this file)")
print(pack.review_type.value_counts().to_string())
