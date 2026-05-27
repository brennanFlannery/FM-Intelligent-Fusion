#!/usr/bin/env python3
"""
Create Survival Labels CSV from Grade Labels

This script converts the grade-based labels CSV into a survival-based labels CSV
by matching slides to patients in the master Excel file and extracting time-to-event
and event indicator data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys


def extract_patient_id(full_id: str) -> Optional[str]:
    """Extract short patient ID from full slide ID format."""
    if pd.isna(full_id) or not isinstance(full_id, str) or not full_id.strip():
        return None
    
    # Extract first three segments separated by dashes
    parts = full_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    
    # Fallback: try regex pattern
    import re
    match = re.match(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)', full_id)
    if match:
        return match.group(1)
    
    return None


def load_grade_labels(csv_path: str) -> pd.DataFrame:
    """Load grade_labels.csv and validate required columns."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Grade labels file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['slide_id', 'case_id', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Grade labels file missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading grade labels file: {e}")


def load_master_file(excel_path: str) -> pd.DataFrame:
    """Load the master Excel file with survival data."""
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Master file not found: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        required_cols = ['PatientID', 'vital_status', 'days_to_last_followup', 'death_days_to']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Master file missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading master file: {e}")


def is_valid_value(value) -> bool:
    """Check if a value is valid (non-empty, non-zero)."""
    if pd.isna(value):
        return False
    
    if isinstance(value, (int, float)):
        return value != 0
    
    if isinstance(value, str):
        stripped = value.strip()
        try:
            num_val = float(stripped)
            return num_val != 0
        except ValueError:
            return len(stripped) > 0
    
    return value is not None


def prepare_survival_data(master_row: pd.Series) -> Optional[Tuple[float, int]]:
    """
    Extract time-to-event and event indicator from master file row.
    
    Returns:
        Tuple of (time_days, event) where event=1 if death, 0 if censored
        Returns None if data is invalid
    """
    death_days = master_row.get('death_days_to', None)
    followup_days = master_row.get('days_to_last_followup', None)
    
    # Check if death_days_to is valid
    if is_valid_value(death_days):
        try:
            time = float(death_days)
            if time > 0:
                return (time, 1)  # Event occurred (death)
        except (ValueError, TypeError):
            pass
    
    # Check if days_to_last_followup is valid
    if is_valid_value(followup_days):
        try:
            time = float(followup_days)
            if time > 0:
                return (time, 0)  # Censored (alive at last followup)
        except (ValueError, TypeError):
            pass
    
    return None


def match_slides_to_master(grade_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Create mapping from slide_id to master file row."""
    # Create mapping from short patient ID to master file row
    master_mapping = {}
    for idx, row in master_df.iterrows():
        patient_id = row['PatientID']
        if pd.notna(patient_id):
            patient_id_str = str(patient_id).strip()
            master_mapping[patient_id_str] = row
    
    # Extract and match slides from grade labels
    slide_to_master = {}
    unmatched_slides = []
    
    for idx, row in grade_df.iterrows():
        slide_id = row['slide_id']
        if pd.isna(slide_id):
            unmatched_slides.append(f"Row {idx}: missing slide_id")
            continue
        
        short_id = extract_patient_id(slide_id)
        if short_id is None:
            unmatched_slides.append(f"{slide_id}: could not extract patient ID")
            continue
        
        if short_id in master_mapping:
            slide_to_master[slide_id] = master_mapping[short_id]
        else:
            unmatched_slides.append(f"{slide_id} (patient: {short_id}): not found in master file")
    
    return slide_to_master, unmatched_slides


def create_survival_labels(grade_df: pd.DataFrame, master_mapping: Dict[str, pd.Series], 
                          filter_invalid: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create survival labels DataFrame from grade labels and master mapping.
    
    Returns:
        Tuple of (survival_df, invalid_slides_list)
    """
    data = []
    invalid_slides = []
    
    for idx, row in grade_df.iterrows():
        slide_id = row['slide_id']
        case_id = row['case_id']
        
        if slide_id not in master_mapping:
            if not filter_invalid:
                invalid_slides.append(f"{slide_id}: not matched to master file")
            continue
        
        master_row = master_mapping[slide_id]
        survival_data = prepare_survival_data(master_row)
        
        if survival_data is None:
            if not filter_invalid:
                invalid_slides.append(f"{slide_id}: invalid survival data")
            continue
        
        time, event = survival_data
        
        # Validate data
        if not isinstance(time, (int, float)) or time <= 0:
            invalid_slides.append(f"{slide_id}: invalid time value {time}")
            if filter_invalid:
                continue
        
        if event not in [0, 1]:
            invalid_slides.append(f"{slide_id}: invalid event value {event}")
            if filter_invalid:
                continue
        
        data.append({
            'slide_id': slide_id,
            'case_id': case_id,
            'time': float(time),
            'event': int(event)
        })
    
    df = pd.DataFrame(data)
    return df, invalid_slides


def save_survival_labels(df: pd.DataFrame, output_path: str):
    """Save survival labels DataFrame to CSV."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert grade labels CSV to survival labels CSV"
    )
    parser.add_argument(
        '--grade_labels',
        type=str,
        default=None,
        help='Path to grade_labels.csv (default: kirc_splits/pre_split/grade_labels.csv relative to script directory)'
    )
    parser.add_argument(
        '--master_xlsx',
        type=str,
        default=None,
        help='Path to master Excel file (default: kca_master_hpc_cptacupdated.xlsx in script directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV (default: kirc_splits/pre_split/survival_labels.csv relative to script directory)'
    )
    parser.add_argument(
        '--filter_invalid',
        action='store_true',
        default=True,
        help='Exclude rows with invalid survival data (default: True)'
    )
    parser.add_argument(
        '--no_filter_invalid',
        action='store_false',
        dest='filter_invalid',
        help='Include rows with invalid survival data (with warnings)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed matching statistics'
    )
    
    args = parser.parse_args()
    
    # Set default paths relative to script directory
    script_dir = Path(__file__).parent
    
    if args.grade_labels is None:
        args.grade_labels = str(script_dir / 'kirc_splits' / 'pre_split' / 'grade_labels.csv')
    if args.master_xlsx is None:
        args.master_xlsx = str(script_dir / 'kca_master_hpc_cptacupdated.xlsx')
    if args.output is None:
        args.output = str(script_dir / 'kirc_splits' / 'pre_split' / 'survival_labels.csv')
    
    try:
        # Load input files
        print("Loading input files...")
        grade_df = load_grade_labels(args.grade_labels)
        master_df = load_master_file(args.master_xlsx)
        print(f"Loaded {len(grade_df)} rows from grade labels file")
        print(f"Loaded {len(master_df)} rows from master file")
        
        # Match slides to master file
        print("\nMatching slides to master file...")
        master_mapping, unmatched = match_slides_to_master(grade_df, master_df)
        print(f"Matched {len(master_mapping)} slides to master file")
        
        if unmatched and args.verbose:
            print(f"\nUnmatched slides ({len(unmatched)}):")
            for slide in unmatched[:10]:  # Show first 10
                print(f"  {slide}")
            if len(unmatched) > 10:
                print(f"  ... and {len(unmatched) - 10} more")
        
        # Create survival labels
        print("\nCreating survival labels...")
        survival_df, invalid_slides = create_survival_labels(
            grade_df, master_mapping, filter_invalid=args.filter_invalid
        )
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Total slides processed:        {len(grade_df)}")
        print(f"Successfully matched slides:    {len(master_mapping)}")
        print(f"Slides with valid survival:    {len(survival_df)}")
        print(f"Unmatched slides:              {len(unmatched)}")
        if invalid_slides:
            print(f"Invalid survival data:         {len(invalid_slides)}")
            if args.verbose:
                print("\nInvalid slides:")
                for slide in invalid_slides[:10]:  # Show first 10
                    print(f"  {slide}")
                if len(invalid_slides) > 10:
                    print(f"  ... and {len(invalid_slides) - 10} more")
        print("=" * 70)
        
        # Validate output data
        if len(survival_df) > 0:
            print(f"\nOutput data validation:")
            print(f"  Time range: {survival_df['time'].min():.2f} - {survival_df['time'].max():.2f} days")
            print(f"  Events (death): {survival_df['event'].sum()} ({survival_df['event'].sum() / len(survival_df) * 100:.1f}%)")
            print(f"  Censored (alive): {(survival_df['event'] == 0).sum()} ({(survival_df['event'] == 0).sum() / len(survival_df) * 100:.1f}%)")
        
        # Save output
        if len(survival_df) == 0:
            print("\nWARNING: No valid survival data found. Output file will be empty.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
        
        print(f"\nSaving output to {args.output}...")
        save_survival_labels(survival_df, args.output)
        print("Done!")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
