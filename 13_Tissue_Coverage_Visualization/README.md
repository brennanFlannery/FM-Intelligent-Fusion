# Group 13 — Tissue Coverage Visualization

Disease-specific scripts for visualizing how well model attention covers the correct tissue types. Files from Kidney and Prostate share the same original name and have been renamed with a disease suffix.

---

### visualize_tumor_normal_coverage_kidney.py
**Purpose:** Read an aggregated Dice/coverage summary CSV (kidney cancer) and produce a multi-panel figure with line plots of attention coverage vs. percentile threshold and ranking bump charts, for tumor and normal tissue types.

**Inputs:**
- `--input` / `--csv` (str, **required**): Aggregated summary CSV produced by `aggregate_dice_results.py`; must contain columns: `model`, `percentile`, `mean_tumor_cov`, `std_tumor_cov`, `n_valid_tumor_cov`, `mean_normal_cov`, `std_normal_cov`, `n_valid_normal_cov`, `mean_pct_attn_in_tumor`, `std_pct_attn_in_tumor`, `n_valid_pct_attn_in_tumor`, `mean_pct_attn_in_normal`, `std_pct_attn_in_normal`, `n_valid_pct_attn_in_normal`
- `--output` (str, optional, default=same directory as input CSV, filename `kidney_tumor_normal_coverage.png`): Output file path
- `--figsize` (str, optional, default=`"18 4.7"`): Figure size as `"W H"` in inches
- `--dpi` (int, optional, default=`150`): Output DPI
- `--format` (str, optional, default=`"png"`, choices: `png`, `pdf`, `both`): Output format

**Outputs:**
- `{output_stem}.png` / `.pdf` — combined figure with line plots (coverage vs. percentile) and bump charts (model rankings) for tumor and normal tissue
- `{output_stem}_doublebar_p{percentile}.png` / `.pdf` — double-bar chart per percentile (tumor bars pointing up in red, normal bars pointing down in green)

**Example:**
```bash
python visualize_tumor_normal_coverage_kidney.py \
    --input /path/DiceResults/dice_summary_all_models.csv \
    --output /path/figures/kidney_tumor_normal_coverage.png \
    --figsize "18 4.7" \
    --dpi 300 \
    --format both
```

---

### visualize_tumor_normal_coverage_prostate.py
**Purpose:** Read an aggregated Dice/coverage summary CSV (prostate cancer) and produce a multi-panel figure with line plots and ranking bump charts, for tumor and benign tissue types.

**Inputs:**
- `--input` / `--csv` (str, **required**): Aggregated summary CSV with columns: `model`, `percentile`, `mean_tumor_cov`, `std_tumor_cov`, `n_valid_tumor_cov`, `mean_benign_cov`, `std_benign_cov`, `n_valid_benign_cov`, `mean_pct_attn_in_tumor`, `std_pct_attn_in_tumor`, `n_valid_pct_attn_in_tumor`, `mean_pct_attn_in_benign`, `std_pct_attn_in_benign`, `n_valid_pct_attn_in_benign`
- `--output` (str, optional, default=same directory as input CSV, filename `prostate_tumor_benign_coverage.png`): Output file path
- `--figsize` (str, optional, default=`"18 4.7"`): Figure size as `"W H"` in inches
- `--dpi` (int, optional, default=`150`): Output DPI
- `--format` (str, optional, default=`"png"`, choices: `png`, `pdf`, `both`): Output format

**Outputs:**
- `{output_stem}.png` / `.pdf` — combined figure with line plots and bump charts for tumor and benign tissue
- `{output_stem}_doublebar_p{percentile}.png` / `.pdf` — double-bar chart per percentile (tumor up in red, benign down in green)

**Example:**
```bash
python visualize_tumor_normal_coverage_prostate.py \
    --input /path/DiceResults/dice_summary_all_models.csv \
    --output /path/figures/prostate_tumor_benign_coverage.png \
    --format both
```

---

### visualize_sra_coverage.py
**Purpose:** Read an SRA (Semantic Region Annotation) coverage summary CSV (rectal cancer) and produce a multi-panel figure with per-tissue-type line plots and ranking bump charts across percentile thresholds.

**Inputs:**
- `--input` / `--csv` (str, **required**): SRA summary CSV (e.g., `dice_summary_all_models_sra.csv`) with columns: `model`, `percentile`, `class` (tissue type name), `mean_coverage`, `std_coverage`, `n_valid`
- `--output` (str, optional, default=same directory as input CSV, filename `sra_coverage_summary.png`): Output file path
- `--figsize` (str, optional, default=`"16 14"`): Figure size as `"W H"` in inches
- `--dpi` (int, optional, default=`150`): Output DPI
- `--format` (str, optional, default=`"png"`, choices: `png`, `pdf`, `both`): Output format
- `--bump_tissues` (str, nargs=`*`, optional, default=`["TUM"]`): Tissue class names to include in the ranking bump chart (e.g., `TUM`, `STR`, `MUC`)

**Outputs:**
- `{output_stem}.png` / `.pdf` — multi-panel figure with one line-plot panel per tissue class (coverage vs. percentile, one line per model) and bump chart panels for the specified `--bump_tissues`

**Example:**
```bash
python visualize_sra_coverage.py \
    --input /path/DiceResults/dice_summary_all_models_sra.csv \
    --output /path/figures/sra_coverage_summary.png \
    --bump_tissues TUM STR MUC \
    --figsize "16 14" \
    --format both
```
