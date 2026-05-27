#!/usr/bin/env python3
"""
Convert clinical XLSX + TRIDENT slide-level .h5 bags → CLAM label sheet.
"""

from pathlib import Path
import pandas as pd
import re

# ──── EDIT THESE THREE PATHS ───────────────────────────────────────────────────
CLIN_PATH   = Path("//mnt/pan/Data7/bxf169/KidneyCancerPathology/kca_master_hpc_cptacupdated.xlsx")
BAG_DIR     = Path("//scratch/users/bxf169/KidneyPathologyData/FM_Outputs/20x_512px_0px_overlap/features_conch_v15")  # root that holds *.h5 bags
OUT_CSV     = Path("//mnt/pan/Data7/bxf169/KidneyCancerPathology/grade_labels.csv")
# ────────────────────────────────────────────────────────────────────────────────

## 1 ─ Load clinical table and build {patient_id → grade_int} mapping
clin = pd.read_excel(CLIN_PATH, engine="openpyxl", usecols=["PatientID", "tumor_grade"])
grade_map = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}
clin["label"] = clin["tumor_grade"].map(grade_map)
clin = clin.dropna(subset=["label"])           # keep only rows with valid grade
pt_to_grade = clin.set_index("PatientID")["label"].to_dict()

## 2 ─ Crawl for .h5 files
bag_paths = list(BAG_DIR.rglob("*.h5"))
rows = []

for p in bag_paths:
    slide_id = p.stem                                    # remove .h5
    # patient id = first 3 tokens of TCGA barcode
    m = re.match(r"(TCGA-[^-]+-[^-]+)", slide_id)
    if not m:
        continue
    patient_id = m.group(1)
    if patient_id not in pt_to_grade:
        continue                                         # skip slides w/o grade
    rows.append({"slide_id": slide_id,
                 "label": int(pt_to_grade[patient_id])})

## 3 ─ Save CSV
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Wrote {len(rows)} rows → {OUT_CSV.resolve()}")
