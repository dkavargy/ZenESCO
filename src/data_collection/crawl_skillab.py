"""
ZenESCO - SKILLAB Tracker crawler (industrial occupations, batch mode)

- Target population: 65 ESCO occupations in 12 clusters (TARGETS below),
  aligned with the scope of Computers in Industry. No software developers.
- Crawls one cluster at a time, gently, so the Tracker is not overloaded.
- Keywords are only the retrieval net; every job is flagged target_match = True/False
  depending on whether its Tracker `occupations` field contains a target ESCO occupation.
- Keeps ALL fields returned by /api/jobs, plus mapped skill / occupation labels.
- ONE output file (OUT_FILE) holds everything: it is re-saved every 1,000 new records
  and at the end. Saving goes through a temporary file, so a crash never corrupts it.
- Rerunning resumes from the same file (already-crawled pages and jobs are skipped).

Setup:
    pip install requests tqdm
    set the environment variables SKILLAB_TRACKER_USER and SKILLAB_TRACKER_PASSWORD
    (access to the SKILLAB Tracker is granted by the SKILLAB consortium; no credentials are in this file)

Run:
    python crawl_skillab.py 1 2      # crawl clusters 1 and 2 only
    python crawl_skillab.py          # crawl the default BATCH below
"""

import os
import sys
import json
import time
import datetime as dt
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================
# CONFIG
# ============================

API = "https://skillab-tracker.csd.auth.gr/api"
# Credentials are NEVER stored in the code. Set them as environment variables before running:
#   Windows (cmd):  set SKILLAB_TRACKER_USER=...      set SKILLAB_TRACKER_PASSWORD=...
#   Linux / macOS:  export SKILLAB_TRACKER_USER=...   export SKILLAB_TRACKER_PASSWORD=...
USER = os.environ.get("SKILLAB_TRACKER_USER", "")
PASS = os.environ.get("SKILLAB_TRACKER_PASSWORD", "")

SOURCES = [
    'brightminds', 'jobbguru', 'jobbguru.se', 'jobbland', 'jobbland.se',
    'jobmedic', 'jobmedic.co.uk', 'jobscentral', 'jobs.de', 'kariera.fr',
    'kariera.gr', 'lesjeudis', 'lesjeudis.com', 'OJA'
]

# Base folder of the project data (same BASE as all stage scripts); override with ZENESCO_BASE.
# Keep it outside cloud-synchronised folders, which lock big files while they upload them.
BASE = Path(os.environ.get("ZENESCO_BASE", r"C:\skillab_data"))
OUT_FILE = BASE / "skillab_industrial_jobs.json"
SAVE_EVERY = 5000      # re-save the file every 5,000 new records (the file is big now)
MAX_PAGES = None       # None = take all available pages; e.g. 10 for a quick test
PAGE_SLEEP = 1.0       # seconds between page requests
REQUEST_TIMEOUT = 300  # seconds to wait for one Tracker answer
KW_PER_REQUEST = 3     # keywords sent per query; small queries are much lighter for the Tracker
CLUSTER_SLEEP = 30     # seconds between clusters
LABEL_BATCH = 50       # ids per label request
ALT_PER_OCC = 3        # ESCO alternative labels used as extra keywords per occupation
BATCH = [1, 2, 3]      # default clusters if none are given on the command line

# Multilingual seed keywords per cluster (DE / FR / SV / EL portals).
# English ESCO labels are added automatically from TARGETS.
SEED_KEYWORDS = {
    1: ["Wirtschaftsingenieur", "Produktionsingenieur", "Fertigungsingenieur", "Verfahrensingenieur",
        "ingénieur industriel", "ingénieur production", "ingénieur méthodes", "ingénieur procédés",
        "industriingenjör", "produktionsingenjör", "processingenjör",
        "βιομηχανικός μηχανικός", "μηχανικός παραγωγής"],
    2: ["PLC programmer", "SCADA engineer", "control systems engineer",
        "Automatisierungsingenieur", "Automatisierungstechniker", "ingénieur automaticien",
        "automationsingenjör", "μηχανικός αυτοματισμού"],
    3: ["robot programmer", "Mechatroniker", "Robotik Ingenieur", "ingénieur robotique",
        "ingénieur mécatronique", "mekatronikingenjör", "robottekniker"],
    4: ["reliability engineer", "predictive maintenance", "Instandhaltungsingenieur",
        "Instandhaltungsleiter", "ingénieur maintenance", "underhållsingenjör",
        "μηχανικός συντήρησης"],
    5: ["quality control engineer", "supplier quality engineer", "Qualitätsingenieur",
        "Qualitätsmanager", "ingénieur qualité", "kvalitetsingenjör", "μηχανικός ποιότητας"],
    6: ["smart manufacturing", "additive manufacturing", "Elektroingenieur", "Elektronikingenieur",
        "ingénieur électronique", "ingénieur électricien", "elingenjör", "elektronikingenjör",
        "ηλεκτρολόγος μηχανικός"],
    7: ["product development engineer", "Konstrukteur", "Entwicklungsingenieur", "Maschinenbauingenieur",
        "ingénieur conception", "ingénieur mécanique", "konstruktör", "maskiningenjör",
        "μηχανολόγος μηχανικός"],
    8: ["CAD designer", "PLM engineer", "product lifecycle management", "technical writer",
        "Technischer Zeichner", "Technischer Redakteur", "dessinateur projeteur",
        "rédacteur technique", "CAD-konstruktör", "teknisk skribent"],
    9: ["production manager", "plant manager", "production planner", "Produktionsleiter",
        "Werkleiter", "Produktionsplaner", "responsable de production", "directeur d'usine",
        "produktionschef", "διευθυντής παραγωγής"],
    10: ["supply chain planner", "demand planner", "sales and operations planning",
         "Supply Chain Manager", "Disponent", "responsable supply chain", "supply chain planerare"],
    11: ["logistics manager", "Logistikingenieur", "Logistikleiter", "Lagerleiter",
         "responsable logistique", "logistikchef", "lagerchef", "υπεύθυνος αποθήκης"],
    12: ["procurement manager", "strategic buyer", "Einkaufsleiter", "Einkäufer",
         "acheteur industriel", "inköpschef", "inköpare", "υπεύθυνος προμηθειών"],
}


# ============================
# TARGET OCCUPATIONS
# ============================

# 65 target ESCO occupations in 12 clusters (from the ESCO occupation file).
# label = ESCO preferred label, isco = ISCO-08 unit group, alt = first ESCO alternative labels.
TARGETS = {1: {'slug': 'industrial_manufacturing_engineering',
     'label': 'Industrial and manufacturing engineering',
     'occupations': [{'label': 'industrial engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/53d2052f-edc0-4f44-8be1-164526412bfe',
                      'alt': ['industry 4.0 specialist',
                              'industrial engineers',
                              'industrial technology engineering consultant']},
                     {'label': 'manufacturing engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/6818c837-072a-4120-b913-bd360b7b14d0',
                      'alt': ['manufacturing engineering adviser',
                              'manufacturing engineer',
                              'manufacturing engineering consultant']},
                     {'label': 'production engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/d7380260-1386-4e36-98b5-8115ca4a3536',
                      'alt': ['production engineering expert',
                              'production engineering consultant',
                              'agricultural production engineer']},
                     {'label': 'process engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/b5eaf231-77ad-4a86-8e54-15cc4398aad2',
                      'alt': ['processing engineer', 'process engineer', 'manufacturing systems engineer']},
                     {'label': 'industrial engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/bcc21c63-7eee-4520-8fa7-43eefd389668',
                      'alt': ['industrial tool design inspector',
                              'industrial tool development  technologist',
                              'industrial tool inspector']},
                     {'label': 'process engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/53be4843-8f31-4645-b25c-9a5bbb04f6b6',
                      'alt': ['production supervisor',
                              'resident inspector',
                              'manufacturing  engineering supervisor']},
                     {'label': 'production engineering technician',
                      'isco': '3115',
                      'uri': 'http://data.europa.eu/esco/occupation/11833a96-fc27-4b03-a53d-892ac891a2bb',
                      'alt': ['production engineering instrument mechanic',
                              'production engineering test inspector',
                              'production engineering inspection mechanic']}]},
 2: {'slug': 'automation_control',
     'label': 'Automation, instrumentation and control',
     'occupations': [{'label': 'automation engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/bb609566-3ab6-44dd-8f48-cf0b15b96827',
                      'alt': ['engineer in automated systems',
                              'engineer in automation systems',
                              'engineer in automation']},
                     {'label': 'automation engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/5791a961-4717-4230-a245-a4082a675f04',
                      'alt': ['technician in automated systems',
                              'technologist in automated systems',
                              'automation inspector']},
                     {'label': 'instrumentation engineer',
                      'isco': '2152',
                      'uri': 'http://data.europa.eu/esco/occupation/e6800524-eec2-4c33-af0e-6f36d1d679bc',
                      'alt': ['autonomy engineer',
                              'mine instrumentation engineer',
                              'instrumentation engineer technology engineering specialist']},
                     {'label': 'instrumentation engineering technician',
                      'isco': '3114',
                      'uri': 'http://data.europa.eu/esco/occupation/2bb8adee-2b84-4d7b-84f9-0620bc09ada8',
                      'alt': ['control and instrumentation technologist',
                              'C&I technologist',
                              'C&I engineering technician']},
                     {'label': 'commissioning engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/76abbb82-c103-4d7a-a4c0-14dba4d6199a',
                      'alt': ['installation and commissioning engineer',
                              'test and commissioning engineer',
                              'commissioning engineer machines']},
                     {'label': 'commissioning technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/08f5481e-9233-4ef2-9e4b-945224879ce5',
                      'alt': ['senior commissioning technician']}]},
 3: {'slug': 'robotics_mechatronics',
     'label': 'Robotics and mechatronics',
     'occupations': [{'label': 'robotics engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/551a6797-3ca9-41fc-9dca-9a982a8fd8a6',
                      'alt': ['robotic engineer', 'robotics engineering consultant', 'engineer of robotics']},
                     {'label': 'robotics engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/7833d5cd-873d-4fdd-b2f8-9762d68494a7',
                      'alt': ['technician in robotic systems',
                              'technician in robotics engineering',
                              'robotics engineering technologist']},
                     {'label': 'mechatronics engineer',
                      'isco': '2144',
                      'uri': 'http://data.europa.eu/esco/occupation/ab360abd-32e2-4e03-967d-a10758efffa7',
                      'alt': ['robotics engineer',
                              'engineer of mechatronics',
                              'specialist mechatronic engineer']},
                     {'label': 'mechatronics engineering technician',
                      'isco': '3115',
                      'uri': 'http://data.europa.eu/esco/occupation/edf2e989-d7c5-496e-b365-81fc5cb9eb39',
                      'alt': ['mechatronic device technician',
                              'mechatronics technician',
                              'technologist in mechatronics']},
                     {'label': 'industrial robot controller',
                      'isco': '3139',
                      'uri': 'http://data.europa.eu/esco/occupation/5001506f-dd7d-4801-92ac-6f7e93b45133',
                      'alt': ['industrial robot operative',
                              'delta robot controller',
                              'robotic equipment operative']},
                     {'label': 'automated assembly line operator',
                      'isco': '3139',
                      'uri': 'http://data.europa.eu/esco/occupation/869e3a85-6e4c-4e3c-b14f-a8399c1abd2d',
                      'alt': ['production line maintenance worker',
                              'automated assembly line team member',
                              'automated assembly line operative']}]},
 4: {'slug': 'maintenance_reliability',
     'label': 'Maintenance and reliability',
     'occupations': [{'label': 'maintenance and repair engineer',
                      'isco': '2141',
                      'uri': 'http://data.europa.eu/esco/occupation/615920c5-4f63-4eb3-8b60-afaaed3ab1ff',
                      'alt': ['mechanical engineer', 'equipment engineer', 'maintenance engineer']},
                     {'label': 'predictive maintenance expert',
                      'isco': '2152',
                      'uri': 'http://data.europa.eu/esco/occupation/8edd8bff-cb59-4c9c-ab0f-59e77d18be48',
                      'alt': ['predictive maintenance engineer',
                              'factories predictive maintenance expert',
                              'cars predictive maintenance expert']},
                     {'label': 'industrial maintenance supervisor',
                      'isco': '3115',
                      'uri': 'http://data.europa.eu/esco/occupation/a9b20949-897b-481f-8642-d1f69263baff',
                      'alt': ['maintenance machining and assembly supervisor',
                              'maintenance machine shop supervisor',
                              'leading maintenance machinist']},
                     {'label': 'dependability engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/07dfff6d-a0d0-4202-a697-fe9176c15999',
                      'alt': ['RAMS engineer', 'dependability and security engineer', 'RAM engineer']}]},
 5: {'slug': 'quality_testing',
     'label': 'Quality, testing and metrology',
     'occupations': [{'label': 'quality engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/30c0943c-caa1-411d-b273-26f475971273',
                      'alt': ['quality engineer', 'quality assurance engineer', 'battery quality engineer']},
                     {'label': 'quality engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/998a0bf4-b1be-42ab-aee5-0096c9be3319',
                      'alt': ['quality\xa0technologist', 'quality engineering technician', 'quality expert']},
                     {'label': 'test engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/c8fa93eb-7c2c-42c3-b135-c2e825a6615e',
                      'alt': ['battery validation engineer',
                              'battery test engineer',
                              'agricultural equipment test engineer']},
                     {'label': 'industrial quality manager',
                      'isco': '1321',
                      'uri': 'http://data.europa.eu/esco/occupation/fdb16700-109e-4d02-949e-3213f278df93',
                      'alt': ['engineering and quality control manager',
                              'industrial quality assessment manager',
                              'industrial quality supervisor']},
                     {'label': 'calibration technician',
                      'isco': '3113',
                      'uri': 'http://data.europa.eu/esco/occupation/c332af81-4481-4494-9786-aea1f5c5ffcf',
                      'alt': ['instrument calibration technician',
                              'technician in instrument calibration',
                              'technician in calibration of electrical and electronic equipment']},
                     {'label': 'metrology technician',
                      'isco': '3111',
                      'uri': 'http://data.europa.eu/esco/occupation/e6a3ce2e-c0f2-47a7-937a-61ae9abfe1fa',
                      'alt': ['metrology technologist', 'instrument specialist', 'metrologist']}]},
 6: {'slug': 'smart_manufacturing_electrical',
     'label': 'Smart manufacturing and electrical/electronic engineering',
     'occupations': [{'label': 'microelectronics smart manufacturing engineer',
                      'isco': '2152',
                      'uri': 'http://data.europa.eu/esco/occupation/82616669-cc82-4455-9332-a2df209e9f3e',
                      'alt': ['microelectronics technology consultant',
                              'microelectronics manufacturing engineer',
                              'microelectronics technology engineer']},
                     {'label': 'metal additive manufacturing operator',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/418f3a43-a6b2-4101-aaa0-832fe7d82885',
                      'alt': ['metal AM technician',
                              'AM operator',
                              'metal additive manufacturing technician']},
                     {'label': 'sensor engineer',
                      'isco': '2152',
                      'uri': 'http://data.europa.eu/esco/occupation/4f4d33a1-0b41-4dd9-8a42-aeba5a3974ab',
                      'alt': ['engineer specialised in sensor technology',
                              'engineer in sensor technology',
                              'sensor technology engineering consultant']},
                     {'label': 'electronics engineer',
                      'isco': '2152',
                      'uri': 'http://data.europa.eu/esco/occupation/4d34a9ee-2653-4da0-bef7-f59294577224',
                      'alt': ['electronic technology engineering consultant',
                              'electro engineer',
                              'electronics engineering expert']},
                     {'label': 'electromechanical engineer',
                      'isco': '2151',
                      'uri': 'http://data.europa.eu/esco/occupation/77abfaec-a250-4765-95fa-6091e8da1bba',
                      'alt': ['mechanicals engineer',
                              'electromechanical assembly engineer',
                              'electromechanical technology engineering specialist']},
                     {'label': 'electrical engineer',
                      'isco': '2151',
                      'uri': 'http://data.europa.eu/esco/occupation/86ca306c-ab99-420a-9e2a-aa73c5c4de22',
                      'alt': ['electrical engineer',
                              'electrical engineering consultant',
                              'electrical technology engineering expert']}]},
 7: {'slug': 'product_design_engineering',
     'label': 'Product design and development engineering',
     'occupations': [{'label': 'design engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/6fc8f605-98b9-4218-b5e2-91c4c9c55c4d',
                      'alt': ['product engineer', 'industrial specialist', 'industrial expert']},
                     {'label': 'mechanical engineer',
                      'isco': '2144',
                      'uri': 'http://data.europa.eu/esco/occupation/579254cf-6d69-4889-9000-9c79dc568644',
                      'alt': ['mechanical engineering expert',
                              'mechanics engineer',
                              'mechanical technology engineering consultant']},
                     {'label': 'tooling engineer',
                      'isco': '2144',
                      'uri': 'http://data.europa.eu/esco/occupation/79fed799-ab3a-43d9-bd91-414c2c3b2f57',
                      'alt': ['tooling technology engineering adviser',
                              'tooling engineering adviser',
                              'tooling technology engineering expert']},
                     {'label': 'product development engineering technician',
                      'isco': '3119',
                      'uri': 'http://data.europa.eu/esco/occupation/42381741-f3a7-4b26-b35c-da6fad9f1cfd',
                      'alt': ['product development scientist',
                              'product development engineer',
                              'product development conservation engineer']},
                     {'label': 'industrial designer',
                      'isco': '2163',
                      'uri': 'http://data.europa.eu/esco/occupation/ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775',
                      'alt': ['product design specialist', 'product designer', 'CAD design expert']},
                     {'label': 'application engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/fde4372d-8293-42de-afbb-8b5a9b274fe7',
                      'alt': ['application engineering consultant',
                              'application engineering specialist',
                              'applied research engineer']}]},
 8: {'slug': 'engineering_data_documentation',
     'label': 'Engineering data, CAD and technical documentation',
     'occupations': [{'label': 'computer-aided design operator',
                      'isco': '3118',
                      'uri': 'http://data.europa.eu/esco/occupation/62979364-6fac-41f2-87ca-202bca19a6ab',
                      'alt': ['CAM drafter', '3D CAM technician', 'computer aided designer']},
                     {'label': 'drafter',
                      'isco': '3118',
                      'uri': 'http://data.europa.eu/esco/occupation/a4869ae9-2dfe-48cc-bd97-f803880f3407',
                      'alt': ['draftsperson', 'draughtsperson', 'drafting expert']},
                     {'label': 'technical communicator',
                      'isco': '2641',
                      'uri': 'http://data.europa.eu/esco/occupation/8cb59bc4-1f39-477e-beea-effed12d186c',
                      'alt': ['scientific communicator',
                              'technical author',
                              'advanced technical communicator']},
                     {'label': 'component engineer',
                      'isco': '2149',
                      'uri': 'http://data.europa.eu/esco/occupation/1f958638-066f-470c-9549-a9070ba280a6',
                      'alt': ['component engineering specialist',
                              'components engineers',
                              'component technology engineering consultant']}]},
 9: {'slug': 'production_management',
     'label': 'Production management and planning',
     'occupations': [{'label': 'manufacturing manager',
                      'isco': '1321',
                      'uri': 'http://data.europa.eu/esco/occupation/6426ada1-4d8c-4612-99e3-91bc569805ee',
                      'alt': ['manufacturing plant manager',
                              'manufacturing production manager',
                              'industrial production supervisor']},
                     {'label': 'industrial production manager',
                      'isco': '1321',
                      'uri': 'http://data.europa.eu/esco/occupation/eb9479c6-5fdf-4f45-ace4-28b547b8c7e5',
                      'alt': ['area production manager',
                              'production manufacturing supervisor',
                              'production manufacturing manager']},
                     {'label': 'operations manager',
                      'isco': '1321',
                      'uri': 'http://data.europa.eu/esco/occupation/c6bd511a-d966-4df9-a48e-4f800354f268',
                      'alt': ['operations general manager',
                              'operations supervisor',
                              'operations administrator']},
                     {'label': 'production supervisor',
                      'isco': '3122',
                      'uri': 'http://data.europa.eu/esco/occupation/40517ed4-2769-4283-a15b-4b0d939bd2d3',
                      'alt': ['consumer goods production controller',
                              'consumer goods production foreman',
                              'consumer goods production forewoman']},
                     {'label': 'industrial assembly supervisor',
                      'isco': '3122',
                      'uri': 'http://data.europa.eu/esco/occupation/b12b14fd-5b74-40c6-afd9-dd35b34e27ed',
                      'alt': ['quality control supervisor', 'assembly team leader', 'assembly foreman']},
                     {'label': 'lean manager',
                      'isco': '2421',
                      'uri': 'http://data.europa.eu/esco/occupation/c9862fdd-53b7-4da9-b742-4302d704836c',
                      'alt': ['lean engineer', 'lean facilitator', 'continuous improvement manager']},
                     {'label': 'manufacturing cost estimator',
                      'isco': '2421',
                      'uri': 'http://data.europa.eu/esco/occupation/219d1c30-3849-4de3-b0d6-07da187b6521',
                      'alt': ['production cost estimator',
                              'manufacturing cost recovery technician',
                              'estimator']}]},
 10: {'slug': 'supply_chain',
      'label': 'Supply chain planning',
      'occupations': [{'label': 'supply chain manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/aacc3918-b5d3-484b-9480-5d29aa550d74',
                       'alt': ['supply chain procurement manager',
                               'supply chain project manager',
                               'supply chain specialist']},
                      {'label': 'forecast manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/aeb7c6bd-22c9-43c2-8551-a1cc2560584f',
                       'alt': ['analytic and forecasting manager',
                               'demand planning manager',
                               'forecasting and analysis director']},
                      {'label': 'purchase planner',
                       'isco': '3323',
                       'uri': 'http://data.europa.eu/esco/occupation/dbdc9f7b-31c6-4f21-a9c5-48253bb69c08',
                       'alt': ['purchasing team worker',
                               'graduate supply chain planner',
                               'demand and inventory planner']},
                      {'label': 'logistics analyst',
                       'isco': '2421',
                       'uri': 'http://data.europa.eu/esco/occupation/51cd3a7a-eae6-4dc6-9f24-179d7ec6d230',
                       'alt': ['distribution consultant',
                               'logistics and supply chain analyst',
                               'logistics consultant']},
                      {'label': 'supply chain assistant',
                       'isco': '3343',
                       'uri': 'http://data.europa.eu/esco/occupation/c8b296be-54ae-48cf-be21-031fa8457937',
                       'alt': ['supply chain logistics assistant',
                               'global supply chain assistant',
                               'supply chain supervisor']}]},
 11: {'slug': 'logistics_distribution',
      'label': 'Logistics and distribution',
      'occupations': [{'label': 'logistics engineer',
                       'isco': '2149',
                       'uri': 'http://data.europa.eu/esco/occupation/c4071145-a774-4cf4-aefa-ae18a385cca4',
                       'alt': ['logistics engineering consultant',
                               'logistics technology engineer',
                               'logistics technology engineering expert']},
                      {'label': 'logistics and distribution manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/823db06f-25e1-4658-b063-d6ef956b41f5',
                       'alt': ['logistics and distribution director',
                               'logistics director',
                               'logistics coordinator']},
                      {'label': 'intermodal logistics manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/a2ff6cc3-15fa-4f02-a457-8fd5df7aa29e',
                       'alt': ['manager of intermodal transport logistics',
                               'transportation logistics manager',
                               'manager of intermodal transportation logistics']},
                      {'label': 'warehouse manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/2f5de1ab-eddf-4715-8193-e994912c22ea',
                       'alt': ['warehouse superintendent',
                               'manager of warehouse',
                               'warehouse administration manager']},
                      {'label': 'distribution manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/632e61c8-4e4b-44c7-b213-6dd5a174b143',
                       'alt': ['senior distribution manager',
                               'graduate distribution planner',
                               'trainee logistics manager']}]},
 12: {'slug': 'purchasing_procurement',
      'label': 'Purchasing and procurement',
      'occupations': [{'label': 'purchasing manager',
                       'isco': '1324',
                       'uri': 'http://data.europa.eu/esco/occupation/377865f0-c327-4599-8c7e-d428c6588edf',
                       'alt': ['purchasing coordinator',
                               'purchasing office administrator',
                               'procurement and supply manager']},
                      {'label': 'purchaser',
                       'isco': '3323',
                       'uri': 'http://data.europa.eu/esco/occupation/29fa406d-de89-4515-be8d-0fe7bda993f1',
                       'alt': ['purchasing  office clerk', 'buyer', 'purchasing agent']},
                      {'label': 'procurement support officer',
                       'isco': '3343',
                       'uri': 'http://data.europa.eu/esco/occupation/c9b2b0a7-0adc-4ca7-bf7c-dc95e18aeac7',
                       'alt': ['procurement clerk', 'procurement assistant', 'junior buyer']}]}}


def esco_key(x):
    """Last path segment of an ESCO URI, so full URIs and bare IDs compare equal."""
    return str(x).rstrip("/").split("/")[-1].lower()


def load_clusters():
    clusters = {}
    for n, t in TARGETS.items():
        kws = []
        for o in t["occupations"]:
            kws.append(o["label"])
            kws.extend(o["alt"][:ALT_PER_OCC])
        kws += SEED_KEYWORDS.get(n, [])
        clusters[n] = {"slug": t["slug"], "label": t["label"],
                       "uris": {esco_key(o["uri"]) for o in t["occupations"]},
                       "occupations": [o["label"] for o in t["occupations"]],
                       "keywords": list(dict.fromkeys(k for k in kws if k))}
    return clusters


# ============================
# TRACKER CLIENT
# ============================

class Tracker:
    def __init__(self):
        self.s = requests.Session()
        self.s.verify = False
        self.login()

    def login(self):
        if not USER or not PASS:
            sys.exit("[auth] set SKILLAB_TRACKER_USER and SKILLAB_TRACKER_PASSWORD (see CONFIG)")
        r = self.s.post(f"{API}/login",
                        json={"username": USER, "password": PASS},
                        headers={"accept": "application/json", "Content-Type": "application/json"},
                        timeout=60)
        if r.status_code != 200:
            sys.exit(f"[auth] failed: {r.status_code}")
        token = r.text.replace('"', "")
        self.s.headers.update({"accept": "application/json", "Authorization": f"Bearer {token}"})
        print("[auth] token acquired")

    def post(self, endpoint, data, params=None, retries=5):
        """POST with retries and back-off; returns JSON or None."""
        for attempt in range(1, retries + 1):
            t0 = time.time()
            if endpoint == "jobs":
                tqdm.write(f"  -> requesting page {params.get('page')} (attempt {attempt}) ...")
            try:
                r = self.s.post(f"{API}/{endpoint}", data=data, params=params, timeout=REQUEST_TIMEOUT)
                if endpoint == "jobs":
                    tqdm.write(f"  <- answered in {time.time() - t0:.1f}s (HTTP {r.status_code})")
            except requests.RequestException as e:
                wait = 10 * attempt
                tqdm.write(f"  [net] {e.__class__.__name__}; retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                tqdm.write("  [auth] token expired, logging in again")
                self.login()
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 15 * attempt
                tqdm.write(f"  [http {r.status_code}] retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            tqdm.write(f"  [http {r.status_code}] {r.text[:200]}")
            return None
        tqdm.write("  [fail] giving up on this request")
        return None


# ============================
# SINGLE OUTPUT FILE
# ============================
# Structure of OUT_FILE:
# {
#   "meta":  {"created", "updated", "total_jobs", "clusters": {"1": {"last_page", "jobs", "matches", "done"}}},
#   "jobs":  [ {all Tracker fields..., "mapped_skills", "mapped_occupations", "_crawl": {...}}, ... ]
# }

def load_store():
    """Load the newest valid copy among the output file and its .tmp file."""
    candidates = [OUT_FILE, OUT_FILE.with_suffix(".tmp")]
    candidates = sorted((c for c in candidates if c.exists()), key=lambda c: c.stat().st_mtime, reverse=True)
    for c in candidates:
        try:
            store = json.loads(c.read_text(encoding="utf-8"))
            print(f"[resume] loaded {len(store['jobs'])} jobs from {c}")
            if c != OUT_FILE:
                print(f"[resume] from now on saving to {OUT_FILE}")
            return store
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[resume] skipping {c} ({e.__class__.__name__})")
    now = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[new] starting {OUT_FILE}")
    return {"meta": {"created": now, "updated": now, "total_jobs": 0, "clusters": {}}, "jobs": []}


def save_store(store):
    store["meta"]["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    store["meta"]["total_jobs"] = len(store["jobs"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):                      # antivirus can still lock the file briefly
        try:
            os.replace(tmp, OUT_FILE)              # atomic: the old file is replaced only when complete
            size = OUT_FILE.stat().st_size / 1e6
            tqdm.write(f"  [save] {len(store['jobs'])} jobs in {OUT_FILE} ({size:.1f} MB)")
            return
        except PermissionError:
            tqdm.write(f"  [save] file locked, retrying in 5s ({attempt + 1}/20)")
            time.sleep(5)
    tqdm.write(f"  [save] could not replace {OUT_FILE}; latest data is safe in {tmp}")


# ============================
# LABEL MAPPING (kept in memory, rebuilt from the file on resume)
# ============================

LABELS = {"skills": {}, "occupations": {}}


def rebuild_labels(store):
    for job in store["jobs"]:
        for m in job.get("mapped_skills", []):
            LABELS["skills"][m["id"]] = m["label"]
        for m in job.get("mapped_occupations", []):
            LABELS["occupations"][m["id"]] = m["label"]


def fetch_labels(tr, endpoint, ids):
    cache = LABELS[endpoint]
    missing = [i for i in ids if i not in cache]
    if not missing:
        return
    tqdm.write(f"  [labels] fetching {len(missing)} new {endpoint} labels")
    for i in tqdm(range(0, len(missing), LABEL_BATCH), desc=f"{endpoint} labels", leave=False):
        batch = missing[i:i + LABEL_BATCH]
        res = tr.post(endpoint, data={"ids": batch}, params={"page": 1, "page_size": len(batch)})
        for item in (res or {}).get("items", []):
            if item.get("id") in batch:
                cache[item["id"]] = item.get("label", "Unknown")
        # fallback: one id per request (the method that worked before)
        for b in batch:
            if b in cache:
                continue
            r1 = tr.post(endpoint, data={"ids": b})
            items = (r1 or {}).get("items", [])
            cache[b] = items[0].get("label", "Unknown") if items else "Unknown"
            time.sleep(0.1)
        time.sleep(0.3)


def enrich(tr, records):
    skills = {s for r in records for s in (r.get("skills") or [])}
    occs = {o for r in records for o in (r.get("occupations") or [])}
    fetch_labels(tr, "skills", skills)
    fetch_labels(tr, "occupations", occs)
    for r in records:
        r["mapped_skills"] = [{"id": s, "label": LABELS["skills"].get(s, "Unknown")}
                              for s in (r.get("skills") or [])]
        r["mapped_occupations"] = [{"id": o, "label": LABELS["occupations"].get(o, "Unknown")}
                                   for o in (r.get("occupations") or [])]


def job_id(job):
    return str(job.get("id") or json.dumps(job, sort_keys=True, ensure_ascii=False)[:300])


# ============================
# CRAWL ONE CLUSTER
# ============================

def crawl(tr, num, cluster, store, index):
    slug = cluster["slug"]
    kws = cluster["keywords"]
    groups = [kws[i:i + KW_PER_REQUEST] for i in range(0, len(kws), KW_PER_REQUEST)]
    cmeta = store["meta"]["clusters"].setdefault(
        str(num), {"slug": slug, "jobs": 0, "matches": 0, "done": False, "groups": {}})
    cmeta.setdefault("groups", {})
    if cmeta["done"]:
        print(f"\n=== [{num}] {cluster['label']} already completed; skipping ===")
        return

    print(f"\n=== [{num}] {cluster['label']} | {len(cluster['occupations'])} ESCO occupations | "
          f"{len(kws)} keywords in {len(groups)} small queries | file has {len(store['jobs'])} jobs ===")
    print("  occupations:", "; ".join(cluster["occupations"]))

    pending, first, run_new, run_matched = [], True, 0, 0
    stopped = False

    def flush():
        if pending:
            enrich(tr, pending)
            pending.clear()
        save_store(store)

    for gi, group in enumerate(groups, start=1):
        g = cmeta["groups"].setdefault(str(gi), {"keywords": group, "last_page": 0, "done": False})
        if g["done"]:
            continue
        print(f"\n  [query {gi}/{len(groups)}] keywords: {group}")
        data = {"sources": SOURCES, "keywords": group}
        page, pages_done = g["last_page"] + 1, 0

        while True:
            if MAX_PAGES and pages_done >= MAX_PAGES:
                print(f"  [stop] MAX_PAGES={MAX_PAGES} reached for this query (test mode)")
                break

            res = tr.post("jobs", data=data, params={"page": page})
            if res is None:
                print(f"  [stop] request failed at page {page}; rerun later to resume")
                stopped = True
                break

            items = res.get("items", [])
            if page == g["last_page"] + 1 and pages_done == 0:
                total = res.get("count") or res.get("total")
                if total:
                    print(f"  [info] tracker reports {total} ads for this query")
            if first:
                first = False
                print(f"  [info] response keys: {list(res.keys())}")
                if items:
                    print(f"  [info] job fields: {list(items[0].keys())}")
                    print(f"  [info] example occupations value: {items[0].get('occupations')}")

            if not items:
                print(f"  [done] query {gi} finished after page {page - 1}")
                g["done"] = True
                break

            new = dup = page_matched = 0
            for job in items:
                jid = job_id(job)
                job_occ = {esco_key(o) for o in (job.get("occupations") or [])}
                match = bool(job_occ & cluster["uris"])
                if jid in index:                   # already in the file (other query / cluster / run)
                    c = store["jobs"][index[jid]]["_crawl"]
                    if num not in c["clusters"]:
                        c["clusters"].append(num)
                        page_matched += match
                    if match and num not in c["matched_clusters"]:
                        c["matched_clusters"].append(num)
                    dup += 1
                    continue
                page_matched += match
                job["_crawl"] = {"clusters": [num], "matched_clusters": [num] if match else [],
                                 "query_keywords": group, "first_page": page,
                                 "crawled_at": dt.datetime.now().isoformat(timespec="seconds")}
                index[jid] = len(store["jobs"])
                store["jobs"].append(job)
                pending.append(job)
                new += 1
            run_new += new
            run_matched += page_matched
            g["last_page"] = page
            cmeta["jobs"] += new
            cmeta["matches"] += page_matched
            print(f"  page {page}: {len(items)} items | {new} new | {dup} already in file | "
                  f"{page_matched} target matches | unsaved {len(pending)} | file {len(store['jobs'])}")

            if len(pending) >= SAVE_EVERY:
                flush()

            page += 1
            pages_done += 1
            time.sleep(PAGE_SLEEP)

        if stopped:
            break
        time.sleep(PAGE_SLEEP)

    cmeta["done"] = all(g.get("done") for g in cmeta["groups"].values()) and \
        len(cmeta["groups"]) == len(groups)
    flush()
    print(f"=== [{num}] {'completed' if cmeta['done'] else 'paused'}: {run_new} new jobs this run, "
          f"{run_matched} target matches | file now has {len(store['jobs'])} jobs ===")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    clusters = load_clusters()
    nums = [int(a) for a in sys.argv[1:]] or BATCH
    store = load_store()
    rebuild_labels(store)
    index = {job_id(j): i for i, j in enumerate(store["jobs"])}
    tr = Tracker()
    for i, n in enumerate(nums):
        try:
            crawl(tr, n, clusters[n], store, index)
        except KeyboardInterrupt:
            print("\n[interrupted] saving before exit...")
            save_store(store)
            sys.exit(0)
        if i < len(nums) - 1:
            print(f"[pause] {CLUSTER_SLEEP}s before next cluster")
            time.sleep(CLUSTER_SLEEP)
    print(f"\n[finished] {len(store['jobs'])} jobs in {OUT_FILE}")
