# Group 8 — Dataset Preparation and Label Creation

Scripts for converting raw clinical files and annotations into CLAM-compatible CSVs, splits, and label sheets.

---

### create_clam_csv.py
**Purpose:** Convert a clinical XLSX file and TRIDENT slide-level `.h5` bags into a CLAM-compatible label CSV with slide IDs and integer grade labels.

**Inputs:**
- File: Clinical Excel file (path hardcoded as `CLIN_PATH`) — must contain columns `PatientID` and `tumor_grade` (values: G1–G4)
- File: Directory of `.h5` feature bag files (path hardcoded as `BAG_DIR`), organized by slide ID

**Outputs:**
- CSV at hardcoded `OUT_CSV` path with columns: `slide_id`, `label` (integer 0–3 mapped from G1–G4)

> **Note:** This script has no CLI arguments. Edit `CLIN_PATH`, `BAG_DIR`, and `OUT_CSV` at the top of the file before running.

**Example:**
```bash
python create_clam_csv.py
```

---

### make_clam_splits.py
**Purpose:** Generate k-fold stratified train/val/test split CSVs from a master slide list while maintaining patient-level stratification to prevent leakage.

**Inputs:**
- `--csv_path` (str, **required**): Path to master CSV with columns `case_id`, `slide_id`, `label`
- `--out_dir` (str, **required**): Output directory for split CSVs
- `--k` (int, optional, default=`5`): Number of folds to create
- `--val_frac` (float, optional, default=`0.10`): Fraction of the remaining pool used for validation per fold
- `--test_frac` (float, optional, default=`0.10`): Fraction of patients held out as test set
- `--seed` (int, optional, default=`42`): Random seed for reproducibility

**Outputs:**
- `splits_0.csv` … `splits_{k-1}.csv` in `out_dir`, each with columns `train`, `val`, `test` containing slide IDs

**Example:**
```bash
python make_clam_splits.py \
    --csv_path dataset_csv/grade_labels.csv \
    --out_dir splits/kidney_grade \
    --k 5 \
    --val_frac 0.10 \
    --test_frac 0.10 \
    --seed 42
```

---

### create_splits_seq.py
**Purpose:** Create k-fold classification dataset splits inside the CLAM framework, with optional label fraction support for semi-supervised scenarios.

**Inputs:**
- `--task` (str, **required**, choices: `task_1_tumor_vs_normal`, `task_2_tumor_subtyping`): Task identifier; determines which hardcoded CSV is loaded
- `--k` (int, optional, default=`10`): Number of folds
- `--label_frac` (float, optional, default=`1.0`): Fraction of labels to use (0–1); if ≤ 0, generates splits for [0.1, 0.25, 0.5, 0.75, 1.0] automatically
- `--val_frac` (float, optional, default=`0.1`): Fraction of slides reserved for validation
- `--test_frac` (float, optional, default=`0.1`): Fraction of slides reserved for testing
- `--seed` (int, optional, default=`1`): Random seed

**Outputs:**
- Directory `splits/{task}_{label_frac_pct}/` containing per fold:
  - `splits_{i}.csv` — slide ID lists for train/val/test
  - `splits_{i}_bool.csv` — boolean membership mask
  - `splits_{i}_descriptor.csv` — per-split label distribution summary

**Example:**
```bash
python create_splits_seq.py \
    --task task_1_tumor_vs_normal \
    --k 10 \
    --label_frac 0.5 \
    --seed 1
```

---

### create_clam_labels.py
**Purpose:** Build a CLAM label CSV from TCGA SVS slides and a clinical metadata file, using either Gleason grade or rectal T-stage aggregation logic.

**Inputs:**
- `--slide_folder` (str, **required**): Directory containing `.svs` slide files
- `--clinical_file` (str, **required**): Clinical CSV or XLSX with case IDs and label values
- `--case_id_col` (str, **required**): Column name in the clinical file that identifies the patient/case
- `--target_column` (str, **required**): Clinical column containing raw label values (e.g., grades, stages)
- `--label_type` (str, **required**, choices: `gleason_grade`, `rectal_stage`): Aggregation strategy
  - `gleason_grade` — extracts numeric grades, returns most frequent value per patient
  - `rectal_stage` — parses T-stage strings (T1–T4), returns highest-frequency stage
- `--output_dir` (str, optional, default=parent directory of `--clinical_file`): Where to write the output CSV

**Outputs:**
- CSV at `{output_dir}/{label_type}_clam.csv` with columns: `slide_id`, `case_id`, `label`

**Example:**
```bash
python create_clam_labels.py \
    --slide_folder /data/prostate/slides \
    --clinical_file clinical/tcga_prad_clinical.xlsx \
    --case_id_col PatientID \
    --target_column tumor_grade \
    --label_type gleason_grade \
    --output_dir output/prostate/
```

---

### create_survival_labels.py
**Purpose:** Convert a grade-based label CSV into survival labels (time-to-event + event indicator) by matching slides to a master clinical Excel file.

**Inputs:**
- `--grade_labels` (str, optional, default=`kirc_splits/pre_split/grade_labels.csv`): Grade label CSV with columns `slide_id`, `case_id`, `label`
- `--master_xlsx` (str, optional, default=`kca_master_hpc_cptacupdated.xlsx`): Master Excel file with columns `PatientID`, `vital_status`, `days_to_last_followup`, `death_days_to`
- `--output` (str, optional, default=`kirc_splits/pre_split/survival_labels.csv`): Output CSV path
- `--filter_invalid` / `--no_filter_invalid` (flag, optional, default=filter enabled): Whether to exclude rows with missing or invalid survival data
- `--verbose` (flag, optional): Print detailed matching statistics

**Outputs:**
- CSV at `--output` path with columns: `slide_id`, `case_id`, `time` (float, days), `event` (int: 0=censored/alive, 1=death)

**Example:**
```bash
python create_survival_labels.py \
    --grade_labels kirc_splits/pre_split/grade_labels.csv \
    --master_xlsx kca_master_hpc_cptacupdated.xlsx \
    --output kirc_splits/pre_split/survival_labels.csv \
    --verbose
```

---

### analyze_split_validation.py
**Purpose:** Validate patient splits by checking how many patients have complete vital status and follow-up/death data in the master clinical file.

**Inputs:**
- File: Split CSV (hardcoded path `kirc_splits/splits_0.csv`) with columns `train`, `val`, `test` containing full patient slide IDs
- File: Master Excel file (hardcoded path `kca_master_hpc_cptacupdated.xlsx`) with columns `PatientID`, `vital_status`, `days_to_last_followup`, `death_days_to`

**Outputs:**
- Console report: Per-split (train/val/test) counts of total patients, matched patients, and patients with valid survival data, with percentages

> **Note:** No CLI arguments. Edit the hardcoded paths at lines 249–250 before running.

**Example:**
```bash
python analyze_split_validation.py
```

---

### analyze_annotation_colors.py
**Purpose:** Parse XML annotation files and extract unique `LineColor` values (Windows COLORREF format), converting them to RGB and hex for annotation audit purposes.

**Inputs:**
- `annotations_dir` (str, positional, optional, default=hardcoded CCIPD server path): Directory containing `.xml` annotation files
- `-v` / `--verbose` (flag, optional): Enable verbose per-file output

**Outputs:**
- Console report table with columns: decimal color value, RGB tuple, hex code (`#RRGGBB`), occurrence count, and sample file names

**Example:**
```bash
python analyze_annotation_colors.py /path/to/annotation/xmls --verbose
```
