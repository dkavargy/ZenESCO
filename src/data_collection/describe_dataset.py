"""
ZenESCO - descriptive statistics (reads the small tables made by convert_to_tables.py)

Prints an overview in the console. Nothing is saved. Uses little memory:
the description text is never loaded.

Run (after convert_to_tables.py):
    python describe_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

TABLES = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")) / "tables"
TOP_N = 25

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 200)


def title(text):
    print("\n" + "=" * 80 + f"\n{text}\n" + "=" * 80)


def counts(series, name, total=None):
    t = series.fillna("(missing)").replace({"": "(missing)"}).value_counts().rename_axis(name)
    t = t.reset_index(name="ads")
    if total:
        t["% of ads"] = (t.ads / total * 100).round(1)
    return t


print(f"Loading tables from {TABLES} ...")
ads = pd.read_parquet(TABLES / "ads.parquet")
ad_skills = pd.read_parquet(TABLES / "ad_skills.parquet")
occ = pd.read_parquet(TABLES / "ad_tracker_occ.parquet")
skill_labels = pd.read_parquet(TABLES / "skill_labels.parquet").set_index("skill_uri")["label"]
occ_labels = pd.read_parquet(TABLES / "occ_labels.parquet").set_index("occ_uri")["label"]
N = len(ads)
print(f"{N:,} ads | {len(ad_skills):,} ad-skill rows | {len(occ):,} ad-occupation rows")

# ============================
title("1. OVERVIEW")
ads["is_target"] = ads["matched_clusters"].fillna("") != ""
print(pd.DataFrame({"metric": [
    "ads", "unique ad ids", "ads with a Tracker target match (unreliable, see notes)",
    "unique sources", "ads with 0 skills", "mean skills per ad", "median skills per ad",
    "mean Tracker occupations per ad", "median description length (words)",
    "distinct skills", "distinct Tracker occupations"],
    "value": pd.Series([
        N, ads.ad_id.nunique(), int(ads.is_target.sum()), ads.source.nunique(),
        int((ads.n_skills == 0).sum()), round(ads.n_skills.mean(), 1), int(ads.n_skills.median()),
        round(ads.n_tracker_occ.mean(), 1), int(ads.desc_words.median()),
        ad_skills.skill_uri.nunique(), occ.occ_uri.nunique()], dtype=object)}).to_string(index=False))

title("1b. ADS PER CRAWL CLUSTER (an ad can belong to several)")
cl = ads["clusters"].fillna("").str.split(",").explode()
print(counts(cl[cl != ""], "cluster").sort_values("cluster", key=lambda s: s.astype(int)).to_string(index=False))

# ============================
title("2. MISSING / EMPTY FIELDS")
cols = ["title", "upload_date", "country", "location", "nuts1", "nuts2", "nuts3",
        "experience_level", "type", "sectors", "source"]
miss = pd.DataFrame({"field": cols, "empty_ads": [
    int(ads[c].isna().sum() + (ads[c].astype(str).str.strip().isin(["", "nan", "None", "N/A"])
                               & ads[c].notna()).sum()) for c in cols]})
miss.loc[len(miss)] = ["skills (0 skills)", int((ads.n_skills == 0).sum())]
miss.loc[len(miss)] = ["description (0 words)", int((ads.desc_words == 0).sum())]
miss["empty_%"] = (miss.empty_ads / N * 100).round(1)
print(miss.to_string(index=False))

# ============================
title("3. TIME COVERAGE")
ads["date"] = pd.to_datetime(ads["upload_date"].str[:10], format="%Y-%m-%d", errors="coerce")
print(f"Parsed dates: {ads.date.notna().sum():,} of {N:,} | from {ads.date.min().date()} to {ads.date.max().date()}")
ads["month"] = ads.date.dt.to_period("M").astype(str)
m = ads.groupby("month").size().rename("ads").reset_index()
print(m.to_string(index=False))
core = ads.date.between("2024-07-01", "2025-06-30")
print(f"\nCore window Jul 2024 - Jun 2025: {core.sum():,} ads ({core.mean() * 100:.1f}%)")
ads["half"] = pd.cut(ads.date, [pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01"),
                                pd.Timestamp("2025-07-01")], right=False, labels=["H2 2024", "H1 2025"])
print(ads.groupby(["half", "country"], observed=True).size().unstack(fill_value=0).to_string())

# ============================
title("4. SOURCES")
print(counts(ads.source, "source", N).to_string(index=False))
title("4b. COUNTRY")
print(counts(ads.country, "country", N).to_string(index=False))

# ============================
title("5. EXPERIENCE LEVEL")
print(counts(ads.experience_level, "experience_level", N).head(TOP_N).to_string(index=False))
title("5b. CONTRACT TYPE")
print(counts(ads.type, "type", N).head(TOP_N).to_string(index=False))
title("5c. SECTORS (Tracker labels; an ad can have several)")
sec = ads.sectors.dropna().str.split(r" \| ").explode()
print(f"{sec.nunique()} distinct sectors")
print(counts(sec, "sector", N).head(TOP_N).to_string(index=False))

# ============================
title("6. TOP TRACKER OCCUPATIONS (guesses, not the ad's real occupation)")
esco_occ = occ[occ.occ_uri.str.contains("/esco/occupation/")]
t = esco_occ.occ_uri.value_counts().head(TOP_N).rename_axis("occ_uri").reset_index(name="ads")
t.insert(0, "occupation", t.occ_uri.map(occ_labels))
print(t.drop(columns="occ_uri").to_string(index=False))

# ============================
title("7. TOP SKILLS")
t = ad_skills.skill_uri.value_counts().head(TOP_N).rename_axis("skill_uri").reset_index(name="ads")
t.insert(0, "skill", t.skill_uri.map(skill_labels))
t["% of ads"] = (t.ads / N * 100).round(1)
print(f"{ad_skills.skill_uri.nunique():,} distinct skills")
print(t.drop(columns="skill_uri").to_string(index=False))

title("7b. TOP SKILLS PER COUNTRY (SE, FR)")
sk = ad_skills.merge(ads[["ad_id", "country"]], on="ad_id")
for c in ["SE", "FR"]:
    n_c = (ads.country == c).sum()
    t = sk[sk.country == c].skill_uri.value_counts().head(15).rename_axis("skill_uri").reset_index(name="ads")
    t.insert(0, "skill", t.skill_uri.map(skill_labels))
    t["% of ads"] = (t.ads / n_c * 100).round(1)
    print(f"\n{c} ({n_c:,} ads)")
    print(t.drop(columns="skill_uri").to_string(index=False))

title("7c. SKILLS PER AD")
print(ads.n_skills.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

title("8. TITLES (most frequent; input for occupation assignment)")
tt = ads.title.str.lower().str.strip().value_counts()
print(f"{tt.size:,} distinct titles; top {TOP_N}:")
print(tt.head(TOP_N).rename_axis("title").reset_index(name="ads").to_string(index=False))

print("\nDone.")
