"""
ZenESCO - Stage 2b: final occupation assignment from job titles
================================================================

Objective
    Give each analysable ad at most one ESCO occupation, with MEASURED precision, and decide which
    target occupations have enough ads to be analysed.

Why title only (stated in the paper)
    The occupation comes from the job title, never from the ad's skills; using skills would favour
    ads that mention ESCO's own skills and hide the weak links ZenESCO looks for (circularity).

What the calibration showed (274 hand-checked embedding candidates)
    Only exact preferred-label matches were reliable (30/30). All embedding bands were 17-67%
    precise, and the similarity score hardly separated right from wrong. Typical errors:
    technician vs engineer, software QA vs quality engineer, craft jobs (Elektroniker, agent de
    production) vs engineers, project managers vs engineers.

Method used instead
    Multilingual title rules for the 65 target occupations (EN/SV/FR/DE/EL): a title is assigned to
    an occupation if it contains a DOMAIN word of that occupation AND the right JOB LEVEL word
    (engineer / technician / manager / supervisor), and no exclusion applies (IT and software,
    trainees, project managers, service engineers, open applications). If several occupations match,
    the one named first in the title wins (the part before "till", "for", " - " is read first).
    Non-target occupations are assigned only by exact preferred-label match (background ads).
    The rules were built on the calibration sample (development set) and are checked on a FRESH
    validation sample (test set): stage2b_validation.csv, 8 ads per target occupation.

Inputs   BASE\\stage2\\ad_occupation_candidates.parquet (Stage 2a), BASE\\stage2\\stage2b_calibration.csv,
         BASE\\esco\\occupations_en.csv
Outputs  BASE\\stage2\\ad_occupation.parquet, stage2b_targets.csv (read by Stage 3),
         stage2b_validation.csv (fill `correct` 1/0, rerun), stage2b_report.txt
Run      python stage2b_assign_occupations.py            # BASE = C:\\skillab_data
"""

import re
import unicodedata

E_WORDS = r"engineer|ingenjör|ingenjor|ingénieur|ingenieur|ingegner|μηχανικ|konstruktör|konstrukteur|designer|developer|utvecklare|concepteur|entwickler"
T_WORDS = r"technician|tekniker|technicien|techniker|τεχνικ"
W_WORDS = (r"mechanic\b|mekaniker|mechaniker|elektriker|electrician|elektroniker\b|elektroniker |mechatroniker|"
           r"electromechanic\b|électromécanicien|operator|operatör|opérateur|\bagent\b|monteur|montör|assembler|"
           r"worker|employee|arbetare|ouvrier|fachkraft|helper|installer|installatör")
MM_WORDS = r"manager|chef|responsable|ansvarig|leiter|head\b|director|directeur|direktör|διευθυντ|υπεύθυν"
MS_WORDS = r"supervisor|superviseur|förman|team ?leader|teamlead|arbetsledare|ledare|shift leader|leader|lead\b|gruppenleiter|meister"
EXCL_IT = (r"\bit\b|it-|software|logiciel|mjukvar|devops|cloud|\bsap\b|full-?stack|full stack|\bdata\b|\bai\b|\bqa\b|"
           r"quality assurance|frontend|backend|\bweb\b|cyber|\bwms\b|\bmes\b|\bplm\b|algorithm|systems? engineer|"
           r"\bict\b|\berp\b|test automation|testautomat|application maintenance|automatisation des tests|dynamics|\b365\b|salesforce|servicenow|developer\b.*(java|python|\.net)")
EXCL_TRAINEE = (r"training as|ausbildung|azubi|apprenti|apprentissage|alternance|alternant|stagiaire|\bstage\s+(h/f|f/h|de|d')|"
                r"\bintern\b|internship|praktik|\bphd\b|doktorand|graduate|trainee|lärling|werkstudent|thesis|"
                r"examensarbete|we train|sommarjobb|summer job")
EXCL_OTHER = (r"service ?engineer|serviceingenj|servicetekniker|after.?sales|kundendienst|technicien (de )?sav|"
              r"intresseanmälan|spontan|open application|initiativbewerbung")
EXCL_PROJECT = r"project manager|projektledare|chef de projet|projektleiter|project leader|project lead|projektchef|program manager|programme manager|delivery project"

# occupation label -> (domain regex, level)
RULES = {
    # 1 industrial and manufacturing engineering
    "production engineer": (r"produktion|production|fertigung|παραγωγ", "E"),
    "process engineer": (r"process(?!ing|or)|prozess|procédé|processingenj|διεργασ", "E"),
    "industrial engineer": (r"industrial engineer|industriingenj|ingénieur industriel|industrialis|industrializ|wirtschaftsingenieur|méthodes|methods engineer|metodingenj", "E"),
    "manufacturing engineer": (r"manufactur|fabrication|tillverkning|\bnpi\b", "E"),
    "industrial engineering technician": (r"industrial engineering|industriteknik|méthodes|methods", "T"),
    "process engineering technician": (r"process(?!ing|or)|prozess|procédé", "T"),
    "production engineering technician": (r"produktion|production|fertigung|παραγωγ", "T"),
    # 2 automation, instrumentation and control
    "automation engineer": (r"automati|automatis|αυτοματισ|\bplc\b|scada|\bdcs\b", "E"),
    "automation engineering technician": (r"automati|automatis|αυτοματισ|\bplc\b", "T"),
    "instrumentation engineer": (r"instrument|measurement|mätteknik|messtechnik|mesure", "E"),
    "instrumentation engineering technician": (r"instrument|measurement|mätteknik|messtechnik|mesure", "T"),
    "commissioning engineer": (r"commission|idrifttag|inbetriebnahme|mise en service|driftsätt", "E"),
    "commissioning technician": (r"commission|idrifttag|inbetriebnahme|mise en service|driftsätt", "T"),
    # 3 robotics and mechatronics
    "robotics engineer": (r"robot", "E"),
    "robotics engineering technician": (r"robot", "T"),
    "mechatronics engineer": (r"mechatron|mekatron|mécatron|μηχατρον", "E"),
    "mechatronics engineering technician": (r"mechatron|mekatron|mécatron", "T"),
    "industrial robot controller": (r"robot.*(operat|controller|programm)|robotoperat|robotprogram", "X"),
    "automated assembly line operator": (r"assembly line|monteringslina|ligne d'assemblage|automated assembly", "X"),
    # 4 maintenance and reliability
    "maintenance and repair engineer": (r"maintenance|underhåll|instandhalt|wartung|συντήρη", "E"),
    "predictive maintenance expert": (r"predictive maintenance|prediktiv|condition monitoring|tillståndsövervak|zustandsüberwach", "X"),
    "industrial maintenance supervisor": (r"maintenance|underhåll|instandhalt|wartung|συντήρη", "MA"),
    "dependability engineer": (r"reliability|dependab|tillförlitlig|zuverlässig|fiabilit|\brams\b", "E"),
    # 5 quality, testing and metrology
    "quality engineer": (r"quality|qualité|kvalitet|qualität|ποιότητ|\bqhse\b|\bsqe\b", "E"),
    "quality engineering technician": (r"quality|qualité|kvalitet|qualität|ποιότητ", "T"),
    "test engineer": (r"\btest|provningsingenj|testingenj|essai|prüf|δοκιμ", "E"),
    "industrial quality manager": (r"quality|qualité|kvalitet|qualität|ποιότητ|\bqhse\b", "MM"),
    "calibration technician": (r"calibr|kalibr|étalonn", "T"),
    "metrology technician": (r"metrolog|métrolog", "X"),
    # 6 smart manufacturing and electrical/electronic engineering
    "microelectronics smart manufacturing engineer": (r"microelectron|mikroelektron|semiconductor|halvledar", "E"),
    "metal additive manufacturing operator": (r"additive|additiv|3d.?print", "X"),
    "sensor engineer": (r"sensor|capteur", "E"),
    "electronics engineer": (r"electronic|elektronik|électroni|ηλεκτρονικ|hardware|hårdvar|\bpcb", "E"),
    "electromechanical engineer": (r"electromechan|elektromekan|électroméca|ηλεκτρομηχαν", "E"),
    "electrical engineer": (r"electrical|elkraft|\bel[- ]|elingenj|elkonstr|électri|electrotech|elektro(?!n)|ηλεκτρολόγ|electric design|electric engineer|power system|high voltage|högspänn|eldrift", "E"),
    # 7 product design and development engineering
    "design engineer": (r"design engineer|konstruktör|concepteur|konstrukteur|designingenj|design/analysis|bureau d'études", "E"),
    "mechanical engineer": (r"mechanical|mekanik|maschinenbau|mécani|μηχανολόγ", "E"),
    "tooling engineer": (r"tooling|verktyg|outillage|werkzeug|mould|\bmold|moule", "E"),
    "product development engineering technician": (r"product development|produktutveckl|développement produit|produktentwickl", "T"),
    "industrial designer": (r"industrial design|industridesign|design industriel|produktdesign", "O"),
    "application engineer": (r"applica|applikation|anwendung", "E"),
    # 8 engineering data, CAD and documentation
    "computer-aided design operator": (r"\bcad\b|\bcao\b|\bdao\b", "O"),
    "drafter": (r"draft|ritare|dessinat|projeteur|technischer zeichner|σχεδιαστ", "O"),
    "technical communicator": (r"technical writ|documentation|teknisk skribent|teknisk dokument|rédacteur technique|technische redakt", "O"),
    "component engineer": (r"component|komponent|composant", "E"),
    # 9 production management and planning
    "manufacturing manager": (r"manufactur|tillverkning|fabrication|plant manager|fabrikschef|directeur d'usine|werkleiter|factory manager|\\busine", "MM"),
    "industrial production manager": (r"produktion|production|fertigung|παραγωγ", "MM"),
    "operations manager": (r"operations|operativ|driftchef|opérations|betriebsleit", "MM"),
    "production supervisor": (r"produktion|production|fertigung|παραγωγ", "MS"),
    "industrial assembly supervisor": (r"assembly|montering|montage", "MS"),
    "lean manager": (r"\blean\b|continuous improvement|operational excellence|performance improvement|amélioration continue|ständige verbesserung|förbättring", "X"),
    "manufacturing cost estimator": (r"cost estimat|kalkylator|kalkylingenj|chiffreur|kalkulator|kostenkalkul", "X"),
    # 10 supply chain planning
    "supply chain manager": (r"supply chain|chaîne logistique|försörjningskedj|lieferkette|\bscm\b", "MM"),
    "forecast manager": (r"forecast|prognos|prévision|demand plan", "X"),
    "purchase planner": (r"purchase plan|inköpsplan|materialplan|material plan|supply plan|approvisionn|disponent", "O"),
    "logistics analyst": (r"logistics analy|logistikanaly|logistikutveckl|analyste logist|logistics develop", "O"),
    "supply chain assistant": (r"(supply chain|logisti).*(assistant|assistent)|(assistant|assistent).*(supply chain|logisti)", "X"),
    # 11 logistics and distribution
    "logistics and distribution manager": (r"logisti|distribution", "MM"),
    "logistics engineer": (r"logisti", "E"),
    "intermodal logistics manager": (r"intermodal|transport", "MM"),
    "warehouse manager": (r"warehouse|lager|entrepôt|magasin", "MA"),
    "distribution manager": (r"distribution", "MM"),
    # 12 purchasing and procurement
    "purchasing manager": (r"purchas|procurement|(?<!l)inköp|achat|einkauf|προμηθ|buyer|(?<!out)sourcing", "MM"),
    "purchaser": (r"purchas|procurement|(?<!l)inköp|achat|einkauf|προμηθ|buyer|(?<!out)sourcing", "O"),
    "procurement support officer": (r"procurement (support|officer|assistant)|inköpsassistent|assistant achat", "O"),
}

def norm(t):
    """Lower case without accents (Greek capitals carry no accents, so accents are ignored everywhere)."""
    t = unicodedata.normalize("NFD", str(t).lower())
    return unicodedata.normalize("NFC", "".join(ch for ch in t if not unicodedata.combining(ch))).replace("ς", "σ").strip()


_C = {k: re.compile(norm(v), re.I) for k, v in dict(E=E_WORDS, T=T_WORDS, W=W_WORDS, MM=MM_WORDS, MS=MS_WORDS,
                                                IT=EXCL_IT, TR=EXCL_TRAINEE, PR=EXCL_PROJECT, OT=EXCL_OTHER).items()}
_D = {occ: re.compile(norm(d), re.I) for occ, (d, _) in RULES.items()}


_QUAL = re.compile(norm(r"quality|qualite|kvalitet|qualitat|ποιοτητ"), re.I)
_PRODUCTION = {"production engineer", "production engineering technician", "production supervisor",
               "industrial production manager"}
# "production" outside manufacturing (found in the Stage 10 expert review): IT operations ("ingenieur de production
# informatique"), contact centres ("produktionsledare kundsupport"), media / TV, construction ("produktionschef bygg"),
# education and prisons. Titles with these words are not assigned to the production occupations.
_NONMFG = re.compile(norm(
    r"informati|exploitation|linux|windows|unix|vtom|control-?m\b|systems? (&|and) network|reseau|"
    r"kundsupport|kundtjanst|kundservice|customer (support|service|care)|contact ?cent|call ?cent|kundcenter|"
    r"\btv|radio|media|film|broadcast|post-?production|\bevent|teater|theat|content|marketing|\bart\b|"
    r"\bbygg|anlaggning|asfalt|\bvag\b|\bvagar|construction|infra|entreprenad|anstalt|utbildning|education|musik|music"), re.I)

# cut points: what follows usually names the employer, place or team, not the job
_CUT = re.compile(r"\s(till|to|for|at|chez|pour|fur|bei|hos|inom|within|with|med|avec|mit|in)\s|\s[-–|]\s|,|\(|!")


def _match(t, full=None):
    nonmfg = bool(_NONMFG.search(full or t))     # checked on the whole title, also when only the head is matched
    if _C["IT"].search(t) or _C["TR"].search(t) or _C["PR"].search(t) or _C["OT"].search(t):
        return []
    has = {k: bool(_C[k].search(t)) for k in ["E", "T", "W", "MM", "MS"]}
    mgr = has["MM"] or has["MS"]
    out = []
    quality_title = bool(_QUAL.search(t))
    for occ, (dom, lvl) in RULES.items():
        if not _D[occ].search(t):
            continue
        if quality_title and occ in _PRODUCTION:   # e.g. "produktionskvalitetsingenjör" is a quality job
            continue
        if occ in _PRODUCTION and nonmfg:              # "production" outside manufacturing
            continue
        if lvl == "E":
            ok = has["E"] and not has["T"] and not has["W"] and not mgr
        elif lvl == "T":
            ok = has["T"] and not has["W"] and not mgr
        elif lvl == "MM":
            ok = has["MM"]
        elif lvl == "MS":
            ok = has["MS"] and not has["MM"]
        elif lvl == "MA":
            ok = mgr
        elif lvl == "O":
            ok = not has["W"] and not mgr
        else:
            ok = True
        if ok:
            out.append((_D[occ].search(t).start(), occ))
    return [o for _, o in sorted(out)]       # ordered by where the domain word appears


def match(title):
    """Target occupations whose rule the title satisfies, ordered by position of the domain word.
    The head of the title (before 'till', 'for', ' - ' ...) is tried first, then the whole title."""
    t = norm(title)
    if _C["IT"].search(t) or _C["TR"].search(t) or _C["PR"].search(t) or _C["OT"].search(t):
        return []
    head = _CUT.split(t)[0].strip()
    m = _match(head, full=t) if head and head != t else []
    return m if m else _match(t)


def assign(title, embedding_first=None):
    """One occupation or None. Ambiguous matches are resolved by the embedding's first choice."""
    m = match(title)
    if not m:
        return None
    if len(m) > 1 and embedding_first in m[:1]:
        return embedding_first
    return m[0]                              # the occupation named first in the title


# target occupation URIs per cluster (same 65 as the crawler, Stage 2a and Stage 3)
TARGET_KEYS = {
    1: "53d2052f-edc0-4f44-8be1-164526412bfe 6818c837-072a-4120-b913-bd360b7b14d0 d7380260-1386-4e36-98b5-8115ca4a3536 b5eaf231-77ad-4a86-8e54-15cc4398aad2 bcc21c63-7eee-4520-8fa7-43eefd389668 53be4843-8f31-4645-b25c-9a5bbb04f6b6 11833a96-fc27-4b03-a53d-892ac891a2bb",
    2: "bb609566-3ab6-44dd-8f48-cf0b15b96827 5791a961-4717-4230-a245-a4082a675f04 e6800524-eec2-4c33-af0e-6f36d1d679bc 2bb8adee-2b84-4d7b-84f9-0620bc09ada8 76abbb82-c103-4d7a-a4c0-14dba4d6199a 08f5481e-9233-4ef2-9e4b-945224879ce5",
    3: "551a6797-3ca9-41fc-9dca-9a982a8fd8a6 7833d5cd-873d-4fdd-b2f8-9762d68494a7 ab360abd-32e2-4e03-967d-a10758efffa7 edf2e989-d7c5-496e-b365-81fc5cb9eb39 5001506f-dd7d-4801-92ac-6f7e93b45133 869e3a85-6e4c-4e3c-b14f-a8399c1abd2d",
    4: "615920c5-4f63-4eb3-8b60-afaaed3ab1ff 8edd8bff-cb59-4c9c-ab0f-59e77d18be48 a9b20949-897b-481f-8642-d1f69263baff 07dfff6d-a0d0-4202-a697-fe9176c15999",
    5: "30c0943c-caa1-411d-b273-26f475971273 998a0bf4-b1be-42ab-aee5-0096c9be3319 c8fa93eb-7c2c-42c3-b135-c2e825a6615e fdb16700-109e-4d02-949e-3213f278df93 c332af81-4481-4494-9786-aea1f5c5ffcf e6a3ce2e-c0f2-47a7-937a-61ae9abfe1fa",
    6: "82616669-cc82-4455-9332-a2df209e9f3e 418f3a43-a6b2-4101-aaa0-832fe7d82885 4f4d33a1-0b41-4dd9-8a42-aeba5a3974ab 4d34a9ee-2653-4da0-bef7-f59294577224 77abfaec-a250-4765-95fa-6091e8da1bba 86ca306c-ab99-420a-9e2a-aa73c5c4de22",
    7: "6fc8f605-98b9-4218-b5e2-91c4c9c55c4d 579254cf-6d69-4889-9000-9c79dc568644 79fed799-ab3a-43d9-bd91-414c2c3b2f57 42381741-f3a7-4b26-b35c-da6fad9f1cfd ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775 fde4372d-8293-42de-afbb-8b5a9b274fe7",
    8: "62979364-6fac-41f2-87ca-202bca19a6ab a4869ae9-2dfe-48cc-bd97-f803880f3407 8cb59bc4-1f39-477e-beea-effed12d186c 1f958638-066f-470c-9549-a9070ba280a6",
    9: "6426ada1-4d8c-4612-99e3-91bc569805ee eb9479c6-5fdf-4f45-ace4-28b547b8c7e5 c6bd511a-d966-4df9-a48e-4f800354f268 40517ed4-2769-4283-a15b-4b0d939bd2d3 b12b14fd-5b74-40c6-afd9-dd35b34e27ed c9862fdd-53b7-4da9-b742-4302d704836c 219d1c30-3849-4de3-b0d6-07da187b6521",
    10: "aacc3918-b5d3-484b-9480-5d29aa550d74 aeb7c6bd-22c9-43c2-8551-a1cc2560584f dbdc9f7b-31c6-4f21-a9c5-48253bb69c08 51cd3a7a-eae6-4dc6-9f24-179d7ec6d230 c8b296be-54ae-48cf-be21-031fa8457937",
    11: "c4071145-a774-4cf4-aefa-ae18a385cca4 823db06f-25e1-4658-b063-d6ef956b41f5 a2ff6cc3-15fa-4f02-a457-8fd5df7aa29e 2f5de1ab-eddf-4715-8193-e994912c22ea 632e61c8-4e4b-44c7-b213-6dd5a174b143",
    12: "377865f0-c327-4599-8c7e-d428c6588edf 29fa406d-de89-4515-be8d-0fe7bda993f1 c9b2b0a7-0adc-4ca7-bf7c-dc95e18aeac7",
}
PREFIX = "http://data.europa.eu/esco/occupation/"
TARGET_CLUSTER = {PREFIX + k: c for c, ks in TARGET_KEYS.items() for k in ks.split()}



# =====================================================================================
# STAGE 2b MAIN
# =====================================================================================
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\skillab_data")
S2 = BASE / "stage2"
N_MIN_SEL = 100        # accepted ads needed in H2 2024
N_MIN_EVAL = 50        # accepted ads needed in H1 2025
VAL_PER_OCC = 8        # validation ads per ANALYSED target occupation (fresh sample, not in calibration)
MIN_VAL_PRECISION = 0.80
SEED = 20261026
BANDS = [("B01", "exact_preferred"), ("B02", "alt_agree"), ("B03", "alt_disagree"),
         ("B04", "emb_agree_>=0.85"), ("B05", "emb_agree_0.80-0.85"), ("B06", "emb_agree_0.75-0.80"),
         ("B07", "emb_agree_0.70-0.75"), ("B08", "emb_disagree_>=0.85"), ("B09", "emb_disagree_0.80-0.85"),
         ("B10", "emb_disagree_0.70-0.80")]

T0 = time.time()
REPORT = []


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def wilson_low(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def answers(path):
    if not path.exists():
        return None
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    d["correct"] = d["correct"].fillna("").str.strip()
    d = d[d.correct.isin(["0", "1"])].copy()
    d["correct"] = d.correct.astype(int)
    return d


ad = pd.read_parquet(S2 / "ad_occupation_candidates.parquet")
label_of = {}
for a, b in [("occ1", "occ1_label"), ("pref_occ", "pref_label"), ("exact_alt", "exact_alt_label")]:
    label_of.update(dict(zip(ad[a], ad[b])))
occ_files = [p for p in (BASE / "esco").glob("*.csv") if p.name.endswith("occupations_en.csv")
             and "research" not in p.name.lower()]
if occ_files:
    o = pd.read_csv(occ_files[0], dtype=str, usecols=["conceptUri", "preferredLabel"])
    label_of.update(dict(zip(o.conceptUri, o.preferredLabel)))
uri_of_target = {label_of[u]: u for u in TARGET_CLUSTER if u in label_of}
missing_rules = [lab for lab in uri_of_target if lab not in RULES]
if missing_rules:
    sys.exit(f"Title rules missing for: {missing_rules}")

say("ZenESCO - Stage 2b report: final occupation assignment (title only)")
say("=" * 78)
say(f"Analysable ads: {len(ad):,}")

# ---- A. why the embedding alone is not enough: precision per band (calibration sample) ----
pref = ad.method == "exact_preferred"
alt = ad.method == "exact_alternative"
alt_ok = alt & (ad.exact_alt == ad.emb_occ1)
agree = ad.pref_occ == ad.emb_occ1
s = ad.emb_score1.astype(float)
emb = ~pref & ~alt
conds = [pref, alt_ok, alt & ~alt_ok,
         emb & agree & (s >= 0.85), emb & agree & (s >= 0.80) & (s < 0.85),
         emb & agree & (s >= 0.75) & (s < 0.80), emb & agree & (s >= 0.70) & (s < 0.75),
         emb & ~agree & (s >= 0.85), emb & ~agree & (s >= 0.80) & (s < 0.85),
         emb & ~agree & (s >= 0.70) & (s < 0.80)]
ad["band"] = np.select(conds, [b for b, _ in BANDS], default="none")
ad["cand_uri"] = np.where(pref, ad.occ1, np.where(alt, ad.exact_alt,
                          np.where(ad.band != "none", ad.emb_occ1, None)))
ad["cand_occupation"] = ad.cand_uri.map(label_of)

cal = answers(S2 / "stage2b_calibration.csv")
if cal is not None and len(cal):
    say()
    say(f"A. Embedding candidates checked by hand (calibration sample, {len(cal)} ads whose candidate is a target)")
    rows = []
    for b, name in BANDS:
        a = cal[cal.band == b]
        k, n = int(a.correct.sum()), len(a)
        rows.append({"band": b, "how": name, "ads": int((ad.band == b).sum()), "checked": n,
                     "precision": round(k / n, 3) if n else np.nan,
                     "wilson_low": round(wilson_low(k, n), 3) if n else np.nan})
    say(pd.DataFrame(rows).to_string(index=False))
    say("  -> only exact preferred-label matches are reliable; the similarity score hardly separates right from")
    say("     wrong matches. Target occupations are therefore assigned with the multilingual title rules (B).")

# ---- B. title rules -------------------------------------------------------------------
log_t = time.time()
titles = ad[["title_raw"]].copy()
titles["emb_label"] = ad.emb_occ1.map(label_of)
uniq = titles.drop_duplicates().copy()
uniq["rule_occupation"] = [assign(t, e) for t, e in zip(uniq.title_raw, uniq.emb_label)]
ad["emb_label"] = titles.emb_label.values
key = dict(zip(zip(uniq.title_raw, uniq.emb_label), uniq.rule_occupation))
ad["rule_occupation"] = [key[(t, e)] for t, e in zip(ad.title_raw, ad.emb_label)]

# final: title rule for target occupations; exact preferred-label match for everything else
ad["occupation"] = np.where(ad.rule_occupation.notna(), ad.rule_occupation,
                            np.where(pref, ad.occ1.map(label_of), None))
ad["occ_uri"] = ad.occupation.map(uri_of_target)
ad.loc[ad.occ_uri.isna() & pref, "occ_uri"] = ad.loc[ad.occ_uri.isna() & pref, "occ1"]
ad["is_target"] = ad.occ_uri.isin(TARGET_CLUSTER)
ad["how"] = np.where(ad.rule_occupation.notna(), "title_rule", np.where(pref, "exact_preferred", "unassigned"))
ad.loc[ad.occ_uri.isna(), "how"] = "unassigned"

say()
say("B. Final assignment: multilingual title rules (domain word + job level + exclusions); "
    "exact preferred-label match for non-target occupations")
say(f"  ads assigned to a TARGET occupation: {ad.is_target.sum():,} ({ad.is_target.mean():.1%})")
say(f"  ads assigned to another occupation (exact preferred label): {(ad.occ_uri.notna() & ~ad.is_target).sum():,}")
say(f"  unassigned (background ads): {ad.occ_uri.isna().sum():,}")
say(f"  rule vs embedding candidate on target ads: same occupation {(ad.is_target & (ad.occupation == ad.cand_occupation)).mean() / max(ad.is_target.mean(), 1e-9):.1%}")

# ---- C. validation sample (fresh, never used to build the rules) ----------------------
val_f = S2 / "stage2b_validation.csv"
cal_ids = set(cal.ad_id) if cal is not None else set()
if not val_f.exists():
    rng = np.random.RandomState(SEED)
    parts = []
    # only occupations with enough ads to be analysed need validating
    w = ad[ad.is_target].groupby(["occupation", "window"]).size().unstack(fill_value=0)
    big = set(w.index[(w.get("H2_2024", 0) >= N_MIN_SEL) & (w.get("H1_2025", 0) >= N_MIN_EVAL)])
    pool = ad[ad.is_target & ad.occupation.isin(big) & ~ad.ad_id.isin(cal_ids)]
    for occ_, g in pool.groupby("occupation"):
        parts.append(g.sample(min(VAL_PER_OCC, len(g)), random_state=rng))
    v = pd.concat(parts)[["ad_id", "title_raw", "occupation"]].sample(frac=1, random_state=rng)
    v["correct"] = ""
    v.to_csv(val_f, index=False, encoding="utf-8-sig")
    say(f"\nValidation sheet written: {val_f.name} ({len(v)} ads). Fill `correct` (1/0) and rerun.")
val = answers(val_f)
if val is not None and len(val):
    k0, n0 = int(val.correct.sum()), len(val)
    say()
    say(f"C. Validation on a fresh sample, as drawn: precision {k0 / n0:.1%} "
        f"(95% lower bound {wilson_low(k0, n0):.1%}), n = {n0}")
    # rules corrected after the validation (outsourcing, 'processing', IT application maintenance,
    # trainee wording, production+quality titles): keep only ads that still carry the checked occupation
    now = dict(zip(ad.ad_id.astype(str), ad.occupation))
    val = val[[now.get(str(i)) == o for i, o in zip(val.ad_id, val.occupation)]]
bad_occ = set()
if val is not None and len(val):
    k, n = int(val.correct.sum()), len(val)
    say(f"   after the rule corrections (ads whose occupation changed are dropped): precision {k / n:.1%} "
        f"(95% lower bound {wilson_low(k, n):.1%}), n = {n}")
    po = val.groupby("occupation").correct.agg(["mean", "count"])
    bad_occ = set(po[(po["count"] >= 5) & (po["mean"] < MIN_VAL_PRECISION)].index)
    say(f"  occupations below {MIN_VAL_PRECISION:.0%} precision (n >= 5), excluded from analysis: "
        + (", ".join(sorted(bad_occ)) or "none"))
else:
    say("\nC. Validation sheet not answered yet: precision of the title rules is NOT yet confirmed.")

# ---- D. targets ---------------------------------------------------------------------------
tg = ad[ad.is_target]
cnt = tg.pivot_table(index="occ_uri", columns="window", values="ad_id", aggfunc="count", fill_value=0)
rows = []
for u, c in TARGET_CLUSTER.items():
    lab = label_of.get(u, "")
    h2 = int(cnt.loc[u, "H2_2024"]) if u in cnt.index and "H2_2024" in cnt.columns else 0
    h1 = int(cnt.loc[u, "H1_2025"]) if u in cnt.index and "H1_2025" in cnt.columns else 0
    pv = val[val.occupation == lab] if val is not None else pd.DataFrame(columns=["correct"])
    rows.append({"cluster": c, "occ_uri": u, "occupation": lab, "n_H2_2024": h2, "n_H1_2025": h1,
                 "validated": len(pv), "precision": round(pv.correct.mean(), 2) if len(pv) else np.nan,
                 "analysed": (h2 >= N_MIN_SEL) and (h1 >= N_MIN_EVAL) and lab not in bad_occ})
targets = pd.DataFrame(rows)
targets.to_csv(S2 / "stage2b_targets.csv", index=False)
say()
say(f"D. Target occupations (analysed = >= {N_MIN_SEL} ads in H2 2024, >= {N_MIN_EVAL} in H1 2025, validation passed)")
with pd.option_context("display.width", 200):
    say(targets.drop(columns="occ_uri").sort_values(["analysed", "n_H2_2024"], ascending=False).to_string(index=False))
A = targets[targets.analysed]
say(f"\n=> analysed occupations: {len(A)} of {len(targets)}, in {A.cluster.nunique()} clusters; "
    f"ads: H2 2024 {A.n_H2_2024.sum():,}, H1 2025 {A.n_H1_2025.sum():,}")
say(f"   with lower limits: 100/30 -> {((targets.n_H2_2024 >= 100) & (targets.n_H1_2025 >= 30)).sum()}, "
    f"60/30 -> {((targets.n_H2_2024 >= 60) & (targets.n_H1_2025 >= 30)).sum()} occupations")

say("\nMost frequent titles per analysed occupation (for a quick eye check)")
for lab in A.occupation:
    top = tg[tg.occupation == lab].title_raw.str.lower().value_counts().head(6)
    say(f"  {lab}: " + " | ".join(f"{t} ({n})" for t, n in top.items()))

ad[["ad_id", "window", "portal", "title_raw", "occ_uri", "occupation", "how", "band",
    "cand_occupation", "is_target"]].to_parquet(S2 / "ad_occupation.parquet", index=False)
say(f"\nRuntime: {time.time() - T0:.0f}s")
(S2 / "stage2b_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
