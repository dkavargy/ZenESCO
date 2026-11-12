<p align="center">
  <img src="assets/zenesco_logo.png" alt="ZenESCO: Industrial Skill Taxonomy Maintenance with Graph Perturbation and Decision Support" width="620">
</p>

# ZenESCO

**Evidence-based maintenance of ESCO skill–occupation links for industrial occupations, using graph perturbation and decision support.**

ZenESCO checks every link between an occupation and a skill in the ESCO taxonomy against real labour-market demand and proposes one action for it: **Retain**, **Delete**, **Upgrade** (optional ↔ essential), **Add**, or **Expert review**. Demand is measured in online job advertisements collected by the SKILLAB Tracker. Each proposed edit is tested by applying it to the ESCO graph (a *perturbation*) and measuring whether the edited graph describes later job advertisements at least as well as the original one.

This repository contains **the source code only**. It contains no job advertisements, no derived tables, no results and no credentials (see [Data availability](#data-availability)).

Authors: Dimitrios Christos Kavargyris and Lefteris Angelis, School of Informatics, Aristotle University of Thessaloniki, Greece. Developed in the SKILLAB project (Horizon Europe, Grant Agreement No. 101132663).

---

## Contents

1. [What the method does](#what-the-method-does)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Inputs you must provide](#inputs-you-must-provide)
5. [How to run](#how-to-run)
6. [Stages, scripts and outputs](#stages-scripts-and-outputs)
7. [Decision rules](#decision-rules)
8. [Main settings](#main-settings)
9. [Notes on validity](#notes-on-validity)
10. [Data availability](#data-availability)
11. [Citation, funding, licence](#citation-funding-licence)

---

## What the method does

ESCO describes each occupation by a profile of *essential* and *optional* skills. These profiles are written by experts and updated slowly, so some links are no longer demanded and some demanded skills are missing. ZenESCO treats ESCO as a bipartite graph (occupations and skills as nodes, links as edges) and works in nine steps:

| # | Step | What happens |
|---|------|--------------|
| 1 | Collect and clean job advertisements | Ads are retrieved from the SKILLAB Tracker, de-duplicated and audited. |
| 2 | Extract skills and build the empirical graph | Each ad gets one ESCO occupation **from its title only**; its skills are the ESCO skills extracted by the Tracker. |
| 3 | Build the official ESCO graph | Official profiles from ESCO v1.2, with link type, reuse level and protection flags. |
| 4 | Compute evidence per link | Prevalence (Beta–Binomial), specificity (lift), reference levels per occupation, and whether the absence of a skill is informative (power test). |
| 5 | Identify candidate additions | Skills outside the profile that are specific to the occupation and as common as its existing links. |
| 6 | Identify candidate deletions and upgrades | Weak links, after an advertising discount and a check for related skills that imply the link. |
| 7 | Perturb the graph and evaluate each edit | Each edit is applied alone; the change in occupation identification (mean reciprocal rank, MRR) is measured on ads from a later period. |
| 8 | Joint edits and robustness | Bootstrap stability, a joint guard on all accepted edits together, and a sensitivity analysis of the settings. |
| 9 | Generate final recommendations | One recommendation per link with its evidence trail; conflicting cases go to a domain expert. |

Two periods of ads are used: the second half of 2024 gives the evidence, and the first half of 2025 is used for the perturbation test.

## Repository layout

```
ZenESCO/
├── README.md
├── requirements.txt
├── CITATION.cff
├── .gitignore                         blocks data and result files from being committed
├── assets/
│   └── zenesco_logo.png
└── src/
    ├── data_collection/
    │   ├── crawl_skillab.py           retrieve ads from the SKILLAB Tracker (credentials from environment)
    │   ├── convert_to_tables.py       crawl JSON -> small Parquet tables
    │   └── describe_dataset.py        descriptive statistics of the tables
    ├── stage1_prepare_data.py         cleaning, duplicates, quality audit
    ├── stage2a_occupation_candidates.py   occupation candidates from job titles
    ├── stage2b_assign_occupations.py  final occupation per ad, with measured precision
    ├── stage3_official_network.py     official ESCO graph of the analysed occupations
    ├── stage4_evidence.py             evidence per link and per candidate addition
    ├── stage5_6_screening.py          advertising discount, related skills, screening actions
    ├── stage7_perturbation.py         single-edit perturbation test
    ├── stage8_sensitivity.py          sensitivity analysis (re-runs stages 4–9 under other settings)
    ├── stage9_final_adjustment.py     final recommendations and updated profiles
    ├── stage10_export_review_pack.py  file for the domain expert
    ├── stage11_tool_comparison.py     comparison with decision rules of related tools
    ├── proposition1_absence.py        when is "never mentioned" informative? (Proposition 1)
    ├── figures_stage1_6.py            figures for stages 1–6
    ├── figures_results.py             decision map and skill map
    └── figures_change.py              profile change, skill groups, demand over time
```

All stage scripts stay in one folder on purpose: `stage8_sensitivity.py` makes patched copies of the other stage scripts and expects to find them next to itself.

## Requirements

Python 3.8 or later. Install the packages with

```
pip install -r requirements.txt
```

`sentence-transformers` is needed only by Stage 2a (the multilingual model `paraphrase-multilingual-MiniLM-L12-v2`, about 470 MB, is downloaded on first use). `requests`, `tqdm` and `ijson` are needed only for data collection.

## Inputs you must provide

**1. Job advertisements.** The crawler reads from the SKILLAB Tracker API. Access is granted by the SKILLAB consortium. Credentials are read from two environment variables and are never written in the code:

```
set SKILLAB_TRACKER_USER=your_user            (Windows)
set SKILLAB_TRACKER_PASSWORD=your_password
export SKILLAB_TRACKER_USER=your_user         (Linux / macOS)
export SKILLAB_TRACKER_PASSWORD=your_password
```

**2. ESCO v1.2 (English, CSV).** Download the classification from the ESCO portal (<https://esco.ec.europa.eu/en/use-esco/download>) and put these files in `BASE/esco/`:

`occupations_en.csv`, `skills_en.csv`, `occupationSkillRelations_en.csv`, `broaderRelationsSkillPillar_en.csv`, `skillSkillRelations_en.csv`, `skillGroups_en.csv`, `ISCOGroups_en.csv`, `digitalSkillsCollection_en.csv`, `transversalSkillsCollection_en.csv`, `languageSkillsCollection_en.csv`, `greenSkillsCollection_en.csv`

**3. A base folder.** Every script reads and writes under one folder, called `BASE`. The default is `C:\skillab_data`; any other folder can be given as the first argument (`python stage4_evidence.py D:\my_base`). For the crawler, set the environment variable `ZENESCO_BASE`. The scripts create the sub-folders themselves:

```
BASE/
├── skillab_industrial_jobs.json   crawler output
├── tables/                        convert_to_tables.py
├── esco/                          ESCO v1.2 CSV files (you provide)
├── stage1/ … stage11/, stage5b/   one folder per stage
└── figures/
```

## How to run

Run the scripts from `src/` in this order. Each script prints a report and saves it as `stageN_report.txt` in its output folder.

```
# A. data collection
python data_collection/crawl_skillab.py            # or: crawl_skillab.py 1 2   (clusters 1 and 2 only)
python data_collection/convert_to_tables.py
python data_collection/describe_dataset.py

# B. method
python stage1_prepare_data.py
python stage2a_occupation_candidates.py
python stage2b_assign_occupations.py               # see manual step 1
python stage3_official_network.py
python stage4_evidence.py
python stage5_6_screening.py
python stage7_perturbation.py
python stage9_final_adjustment.py
python stage10_export_review_pack.py               # see manual step 2
python stage9_final_adjustment.py                  # again, to apply the expert decisions

# C. robustness, comparison, figures
python stage8_sensitivity.py
python proposition1_absence.py
python stage11_tool_comparison.py
python figures_stage1_6.py
python figures_results.py
python figures_change.py
```

Two steps need a person:

1. **Occupation assignment (Stage 2).** Stage 1 writes `title_label_sample.csv`, a sample of ads to label by hand. The labelled file is saved as `BASE/stage2/stage2b_calibration.csv` and read by Stage 2b. Stage 2b then writes `stage2b_validation.csv`; fill its `correct` column with 1 or 0 and run Stage 2b again to obtain the measured precision.
2. **Expert review (Stage 10).** `stage10_export_review_pack.py` writes `review_pack.csv` with every conflicting case and an audit sample of the automatic changes. The expert's answers are saved as `BASE/stage10/review_results.xlsx` (or `.csv`). Running Stage 10 again exports a second round with all remaining unreviewed changes. Stage 9 applies the expert decisions when the file exists.

## Stages, scripts and outputs

| Script | Reads | Main outputs |
|---|---|---|
| `crawl_skillab.py` | SKILLAB Tracker API | `skillab_industrial_jobs.json` |
| `convert_to_tables.py` | crawl JSON | `tables/ads.parquet`, `ad_text.parquet`, `ad_skills.parquet`, label tables |
| `stage1_prepare_data.py` | `tables/` | `ads_clean.parquet`, `ad_skills_clean.parquet`, cleaning flow, boilerplate candidates |
| `stage2a_occupation_candidates.py` | Stage 1, ESCO | `title_candidates.parquet`, `ad_occupation_candidates.parquet` |
| `stage2b_assign_occupations.py` | Stage 2a, calibration file | `ad_occupation.parquet`, `stage2b_targets.csv`, validation sample |
| `stage3_official_network.py` | ESCO, Stage 2b targets | `official_edges.parquet`, `occupation_profiles.csv` |
| `stage4_evidence.py` | Stages 1–3 | `edge_evidence.parquet`, `addition_candidates.parquet`, `pair_evidence.parquet`, `occupation_evidence.csv` |
| `stage5_6_screening.py` | Stage 4, ESCO skill relations | `screened_links.parquet`, `add_candidates.parquet`, screening table |
| `stage7_perturbation.py` | Stages 1, 2, 4, 6 | `curves.csv`, `edit_effects.parquet` |
| `stage9_final_adjustment.py` | Stages 1–7, review file if present | `R1_all_links.csv`, `R2_occupation_summary.csv`, `updated_profiles.csv`, `ZenESCO_tables.xlsx` |
| `stage10_export_review_pack.py` | Stage 9 | `review_pack.csv`, `review_pack_2.csv` |
| `stage8_sensitivity.py` | all earlier outputs | `sensitivity.csv`, `sensitivity_report.txt`, one folder per scenario |
| `proposition1_absence.py` | Stage 6 | `proposition1_links.csv`, `proposition1_occupations.csv` |
| `stage11_tool_comparison.py` | Stages 1, 2, 4, 6, 9, 10 | `tool_comparison.csv`, `tool_edits.csv` |

The header of each script documents its objective, method, inputs, outputs and settings in full.

## Decision rules

Stage 9 gives each link exactly one recommendation. The first rule that matches is applied.

**Link already in ESCO**

| Condition | Recommendation |
|---|---|
| Protected (safety or legal, transversal, last link of the occupation) | Retain |
| Absence of the skill is not informative (too few ads) | Retain |
| Implied by a supported related skill | Retain |
| Weak and optional; identification not worse after removal; stability ≥ 0.8 | Delete |
| Weak and essential; same conditions | Upgrade → optional |
| Optional but demanded like essential links; same conditions | Upgrade → essential |
| One of the three cases above, but the perturbation test fails or stability is 0.5–0.8 | Expert review |
| Otherwise | Retain |

**Pair not in ESCO (candidate addition)**

| Condition | Recommendation |
|---|---|
| Low semantic plausibility | Expert review |
| Clear gain in identification, stability ≥ 0.8, same direction in the later period | Add (essential or optional) |
| Harms identification beyond the margin | Reject addition |
| Otherwise | Expert review |

A recommendation is *Firm* when stability is at least 0.9 and the direction is the same in the later period; otherwise it is *Tentative*.

## Main settings

All settings are constants at the top of each script and are printed in the stage reports.

| Setting | Value | Used in |
|---|---|---|
| Credibility level for "supported" | 0.95 | Stage 4 |
| Significance level of the power test | 0.05 | Stage 4 |
| Prior strength (Beta prior) | 2 | Stage 4 |
| Weight of optional links in a profile | 0.5 | Stages 7, 9 |
| Stability thresholds (accept, firm) | 0.8, 0.9 | Stage 9 |
| Bootstrap resamples | 200 | Stage 9 |
| Top-K demanded skills for coverage | 20 | Stage 9 |

`stage8_sensitivity.py` varies these settings one at a time and reports how the recommendations and the scores change.

## Notes on validity

- **Occupation from the title only.** An ad's occupation never depends on its skills. Using skills would favour ads that already mention ESCO's skills and hide the weak links the method looks for.
- **The later period is not fully independent.** The first-half-2025 ads are used by the perturbation test that selects the edits (Stages 7 and 9). Scores of the final taxonomy on the same ads are therefore optimistic. The screening result of Stage 6, which does not use these ads, is the stricter comparison.
- **Tool comparison (Stage 11).** The related tools are re-implemented as *decision rules* applied to the same data. They are adaptations, not the original systems, and the expert reference set is drawn from ZenESCO's own candidates.
- **Job ads are not the labour market.** Ads over-state some skills and omit others that are taken for granted. The advertising discount and the protection rules reduce this effect; they do not remove it.

## Data availability

No data are distributed with this code. The job advertisements belong to the SKILLAB Tracker and its source portals and are available only to authorised users. ESCO is published by the European Commission and must be downloaded from the ESCO portal. Files produced by the scripts (Parquet, CSV, Excel, figures, reports) contain information derived from the advertisements and must not be committed to this repository; the `.gitignore` file blocks them.

## Citation, funding, licence

If you use this code, please cite:

> Kavargyris, D. C., & Angelis, L. *ZenESCO: Bringing balance to the ESCO taxonomy through an industrial competence-management and decision-support system.* Manuscript in preparation.

This work was supported by the SKILLAB project, funded by the European Union's Horizon Europe programme under Grant Agreement No. 101132663.

Licence: to be defined by the authors and the SKILLAB consortium. Until a licence file is added, all rights are reserved.
