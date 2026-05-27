#!/usr/bin/env python3
import argparse
import re
from collections import Counter
from pathlib import Path
import pandas as pd


def extract_case_id(filename):
    """
    Extract TCGA case ID from filename. Pattern: TCGA-XX-XXXX
    """
    match = re.match(r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})', filename)
    return match.group(1) if match else None


def parse_stage_label(stage_str):
    """
    Parse strings like 'T3' or 'T4a' into integer 1-4.
    Strips leading 'T' (case-insensitive) and any trailing characters.
    Returns the numeric stage value (1-4) or None if unparsable.
    """
    if not isinstance(stage_str, str):
        return None
    match = re.match(r'^T([1-4])', stage_str, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def get_label_for_case(clinical_df, case_id, case_id_col, label_type, target_column):
    """
    Aggregate labels for a given case.
    """
    subset = clinical_df[clinical_df[case_id_col] == case_id]
    if subset.empty:
        return None

    values = subset[target_column].dropna()
    if values.empty:
        return None

    if label_type == 'gleason_grade':
        numeric = pd.to_numeric(values, errors='coerce').dropna().astype(int)
        if numeric.empty:
            return None
        counts = Counter(numeric)
        highest_freq = counts.most_common(1)[0][1]
        candidates = [val for val, freq in counts.items() if freq == highest_freq]
        return max(candidates)

    elif label_type == 'rectal_stage':
        stages = [parse_stage_label(v) for v in values]
        stages = [s for s in stages if s is not None]
        if not stages:
            return None
        counts = Counter(stages)
        highest_freq = counts.most_common(1)[0][1]
        candidates = [val for val, freq in counts.items() if freq == highest_freq]
        return max(candidates)

    else:
        raise ValueError(f"Unknown label_type '{label_type}'")


def main():
    parser = argparse.ArgumentParser(
        description='Label TCGA SVS slides from clinical data.'
    )
    parser.add_argument('--slide_folder', required=True,
                        help='Path to folder containing .svs files')
    parser.add_argument('--clinical_file', required=True,
                        help='Path to clinical CSV or XLSX file')
    parser.add_argument('--case_id_col', required=True,
                        help='Column name for case ID in clinical file')
    parser.add_argument('--target_column', required=True,
                        help='Clinical column containing label values')
    parser.add_argument('--label_type', required=True,
                        choices=['gleason_grade','rectal_stage'],
                        help='Type of label aggregation to apply')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save output CSV; defaults to clinical file directory')
    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.clinical_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clinical data
    if args.clinical_file.lower().endswith('.csv'):
        clinical_df = pd.read_csv(args.clinical_file)
    else:
        clinical_df = pd.read_excel(args.clinical_file)

    # Discover slides
    slide_paths = Path(args.slide_folder).rglob('*.svs')

    records = []
    for p in slide_paths:
        slide_id = p.stem
        case_id = extract_case_id(slide_id)
        if not case_id:
            continue
        label = get_label_for_case(
            clinical_df,
            case_id,
            args.case_id_col,
            args.label_type,
            args.target_column
        )
        if label is None:
            continue
        records.append({
            'slide_id': slide_id,
            'case_id': case_id,
            'label': label
        })

    # Save output
    out_file = output_dir / f"{args.label_type}_clam.csv"
    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Wrote {len(records)} records to {out_file}")


if __name__ == '__main__':
    main()
