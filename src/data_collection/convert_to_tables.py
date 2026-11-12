"""
ZenESCO - Step 1: convert the big crawl JSON into small analysis tables.

Reads skillab_industrial_jobs.json piece by piece (never the whole file in memory)
and writes Parquet tables to C:\\skillab_data\\tables\\ :

    ads.parquet            one row per ad (no description text; light to load)
    ad_text.parquet        ad id + title + description (only when text is needed)
    ad_skills.parquet      one row per ad x skill (ESCO URI)
    ad_tracker_occ.parquet one row per ad x Tracker occupation guess (comparison only)
    skill_labels.parquet   skill URI -> label
    occ_labels.parquet     occupation URI -> label

Run (once):
    pip install ijson pandas pyarrow
    python convert_to_tables.py
"""

import time
import sys
from pathlib import Path

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
SRC = BASE / "skillab_industrial_jobs.json"
OUT = BASE / "tables"
BATCH = 5000          # ads per write; keeps memory low


def text(v):
    """Lists / dicts / None -> plain string."""
    if v is None:
        return None
    if isinstance(v, list):
        return " | ".join(text(x) or "" for x in v) or None
    if isinstance(v, dict):
        return v.get("label") or v.get("name") or str(v)
    return str(v)


INT_COLS = {"n_skills", "n_tracker_occ", "desc_words"}


class Writer:
    """Appends batches of rows to one Parquet file with a FIXED schema:
    every column is text except the counts, so an all-empty first batch
    (e.g. nuts1) can never lock a column into the wrong type."""

    def __init__(self, path, columns):
        self.path, self.rows = path, 0
        self.columns = columns
        self.schema = pa.schema([(c, pa.int64() if c in INT_COLS else pa.string()) for c in columns])
        self.writer = pq.ParquetWriter(path, self.schema)

    def write(self, rows):
        if not rows:
            return
        df = pd.DataFrame(rows, columns=self.columns)
        for c in self.columns:
            if c in INT_COLS:
                df[c] = df[c].fillna(0).astype("int64")
            else:
                df[c] = df[c].where(df[c].notna(), None).astype(object)
                df[c] = df[c].map(lambda v: None if v is None else str(v))
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self.writer.write_table(table)
        self.rows += len(rows)

    def close(self):
        self.writer.close()


AD_COLS = ["ad_id", "source", "source_id", "title", "upload_date", "country", "location",
           "nuts1", "nuts2", "nuts3", "experience_level", "type", "sectors",
           "n_skills", "n_tracker_occ", "desc_words", "clusters", "matched_clusters"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    size_mb = SRC.stat().st_size / 1e6
    print(f"Reading {SRC} ({size_mb:,.0f} MB) in a streaming way ...")

    w_ads = Writer(OUT / "ads.parquet", AD_COLS)
    w_text = Writer(OUT / "ad_text.parquet", ["ad_id", "title", "description"])
    w_skills = Writer(OUT / "ad_skills.parquet", ["ad_id", "skill_uri"])
    w_occ = Writer(OUT / "ad_tracker_occ.parquet", ["ad_id", "occ_uri"])
    skill_labels, occ_labels = {}, {}
    ads, texts, skills, occs = [], [], [], []
    n, t0 = 0, time.time()

    with open(SRC, "rb") as f:
        for job in ijson.items(f, "jobs.item"):
            jid = str(job.get("id"))
            crawl = job.get("_crawl") or {}
            desc = job.get("description") or ""
            sk = [s for s in (job.get("skills") or []) if s]
            oc = [o for o in (job.get("occupations") or []) if o]
            ads.append({
                "ad_id": jid,
                "source": job.get("source"),
                "source_id": text(job.get("source_id")),
                "title": job.get("title"),
                "upload_date": text(job.get("upload_date")),
                "country": job.get("location_code"),
                "location": text(job.get("location")),
                "nuts1": text(job.get("nuts1")),
                "nuts2": text(job.get("nuts2")),
                "nuts3": text(job.get("nuts3")),
                "experience_level": text(job.get("experience_level")),
                "type": text(job.get("type")),
                "sectors": text(job.get("sectors")),
                "n_skills": len(sk),
                "n_tracker_occ": len(oc),
                "desc_words": len(desc.split()),
                "clusters": ",".join(str(c) for c in crawl.get("clusters", [])),
                "matched_clusters": ",".join(str(c) for c in crawl.get("matched_clusters", [])),
            })
            texts.append({"ad_id": jid, "title": job.get("title"), "description": desc})
            skills.extend({"ad_id": jid, "skill_uri": s} for s in sk)
            occs.extend({"ad_id": jid, "occ_uri": o} for o in oc)
            for m in job.get("mapped_skills") or []:
                skill_labels[m["id"]] = m.get("label")
            for m in job.get("mapped_occupations") or []:
                occ_labels[m["id"]] = m.get("label")
            n += 1

            if n % BATCH == 0:
                w_ads.write(ads); w_text.write(texts); w_skills.write(skills); w_occ.write(occs)
                ads, texts, skills, occs = [], [], [], []
                print(f"  {n:,} ads converted ({time.time() - t0:.0f}s)")

    w_ads.write(ads); w_text.write(texts); w_skills.write(skills); w_occ.write(occs)
    for w in (w_ads, w_text, w_skills, w_occ):
        w.close()
    pd.DataFrame(list(skill_labels.items()), columns=["skill_uri", "label"]).to_parquet(
        OUT / "skill_labels.parquet", index=False)
    pd.DataFrame(list(occ_labels.items()), columns=["occ_uri", "label"]).to_parquet(
        OUT / "occ_labels.parquet", index=False)

    print(f"\nDone in {time.time() - t0:.0f}s: {n:,} ads")
    print(f"  ads.parquet            {w_ads.rows:,} rows")
    print(f"  ad_text.parquet        {w_text.rows:,} rows")
    print(f"  ad_skills.parquet      {w_skills.rows:,} rows")
    print(f"  ad_tracker_occ.parquet {w_occ.rows:,} rows")
    print(f"  skill_labels.parquet   {len(skill_labels):,} skills")
    print(f"  occ_labels.parquet     {len(occ_labels):,} occupations")
    print(f"Tables are in {OUT}")


if __name__ == "__main__":
    main()
