#!/usr/bin/env python3
"""
Patient Split Validation Analysis Script

This script analyzes patient data across splits and master files to determine
how many patients in train/val/test splits have valid vital status and 
follow-up/death data.
"""

import pandas as pd
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_splits_file(csv_path: str) -> pd.DataFrame:
    """
    Load the splits CSV file.
    
    Args:
        csv_path: Path to the splits CSV file
        
    Returns:
        DataFrame with train/val/test columns
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Splits file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['train', 'val', 'test']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Splits file missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading splits file: {e}")


def load_master_file(excel_path: str) -> pd.DataFrame:
    """
    Load the master Excel file.
    
    Args:
        excel_path: Path to the master Excel file
        
    Returns:
        DataFrame with PatientID and required columns
    """
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


def extract_patient_id(full_id: str) -> Optional[str]:
    """
    Extract short patient ID from full ID format.
    
    Extracts the first three segments (e.g., 'TCGA-BP-5006' from 
    'TCGA-BP-5006-01Z-00-DX1.98cb6ac9-bb30-4b71-aa66-0d08f80ecd55').
    
    Args:
        full_id: Full patient ID string
        
    Returns:
        Short patient ID or None if extraction fails
    """
    if pd.isna(full_id) or not isinstance(full_id, str) or not full_id.strip():
        return None
    
    # Extract first three segments separated by dashes
    # Pattern: TCGA-XX-####
    parts = full_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    
    # Fallback: try regex pattern
    match = re.match(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)', full_id)
    if match:
        return match.group(1)
    
    return None


def is_valid_value(value) -> bool:
    """
    Check if a value is valid (non-empty, non-zero).
    
    Args:
        value: Value to check
        
    Returns:
        True if value is valid, False otherwise
    """
    # Handle NaN/None
    if pd.isna(value):
        return False
    
    # Handle numeric values
    if isinstance(value, (int, float)):
        return value != 0
    
    # Handle strings
    if isinstance(value, str):
        stripped = value.strip()
        # Check if it's a string representation of zero
        try:
            num_val = float(stripped)
            return num_val != 0
        except ValueError:
            # Not a number, check if non-empty
            return len(stripped) > 0
    
    # For other types, consider non-None as valid
    return value is not None


def has_valid_vital_data(row: pd.Series) -> bool:
    """
    Check if patient has valid vital data.
    
    Valid means:
    - Valid vital_status AND
    - (Valid days_to_last_followup OR valid death_days_to)
    
    Args:
        row: DataFrame row with patient data
        
    Returns:
        True if patient has valid vital data
    """
    # Check vital_status
    vital_status_valid = is_valid_value(row.get('vital_status', None))
    if not vital_status_valid:
        return False
    
    # Check days_to_last_followup OR death_days_to
    days_followup_valid = is_valid_value(row.get('days_to_last_followup', None))
    death_days_valid = is_valid_value(row.get('death_days_to', None))
    
    return days_followup_valid or death_days_valid


def match_patients(splits_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], list]:
    """
    Match patients between splits and master files.
    
    Args:
        splits_df: DataFrame with splits data
        master_df: DataFrame with master data
        
    Returns:
        Tuple of (patient_mapping dict, unmatched_patients list)
    """
    # Create mapping from short patient ID to master file row
    master_mapping = {}
    for idx, row in master_df.iterrows():
        patient_id = row['PatientID']
        if pd.notna(patient_id):
            # Normalize patient ID (convert to string, strip whitespace)
            patient_id_str = str(patient_id).strip()
            master_mapping[patient_id_str] = row
    
    # Extract and match patients from splits
    patient_mapping = {}
    unmatched_patients = []
    
    # Collect all patient IDs from all splits
    all_patient_ids = set()
    for col in ['train', 'val', 'test']:
        if col in splits_df.columns:
            for patient_id in splits_df[col].dropna():
                all_patient_ids.add(patient_id)
    
    # Match each patient
    for full_id in all_patient_ids:
        short_id = extract_patient_id(full_id)
        if short_id is None:
            unmatched_patients.append(full_id)
            continue
        
        if short_id in master_mapping:
            patient_mapping[full_id] = master_mapping[short_id]
        else:
            unmatched_patients.append(full_id)
    
    return patient_mapping, unmatched_patients


def analyze_splits(splits_df: pd.DataFrame, patient_mapping: Dict[str, pd.Series]) -> Dict[str, Dict[str, int]]:
    """
    Analyze splits to count patients with valid vital data.
    
    Args:
        splits_df: DataFrame with splits data
        patient_mapping: Dictionary mapping full patient IDs to master file rows
        
    Returns:
        Dictionary with counts per split
    """
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits_df.columns:
            results[split_name] = {
                'total': 0,
                'matched': 0,
                'valid': 0
            }
            continue
        
        # Get all patient IDs in this split
        split_patients = splits_df[split_name].dropna().tolist()
        total = len(split_patients)
        
        # Count matched and valid patients
        matched = 0
        valid = 0
        
        for full_id in split_patients:
            if full_id in patient_mapping:
                matched += 1
                row = patient_mapping[full_id]
                if has_valid_vital_data(row):
                    valid += 1
        
        results[split_name] = {
            'total': total,
            'matched': matched,
            'valid': valid
        }
    
    return results


def main():
    """Main function to run the analysis."""
    # File paths
    splits_path = Path(__file__).parent / 'kirc_splits' / 'splits_0.csv'
    master_path = Path(__file__).parent / 'kca_master_hpc_cptacupdated.xlsx'
    
    try:
        # Load files
        print("Loading files...")
        splits_df = load_splits_file(str(splits_path))
        master_df = load_master_file(str(master_path))
        
        print(f"Loaded {len(splits_df)} rows from splits file")
        print(f"Loaded {len(master_df)} rows from master file")
        
        # Match patients
        print("\nMatching patients...")
        patient_mapping, unmatched_patients = match_patients(splits_df, master_df)
        
        # Warn about unmatched patients
        if unmatched_patients:
            print(f"\nWARNING: Found {len(unmatched_patients)} unmatched patients", file=sys.stderr)
            if len(unmatched_patients) <= 20:
                for patient_id in unmatched_patients[:20]:
                    print(f"  - {patient_id}", file=sys.stderr)
                if len(unmatched_patients) > 20:
                    print(f"  ... and {len(unmatched_patients) - 20} more", file=sys.stderr)
            else:
                print(f"  (showing first 20 of {len(unmatched_patients)})", file=sys.stderr)
                for patient_id in unmatched_patients[:20]:
                    print(f"  - {patient_id}", file=sys.stderr)
        
        # Analyze splits
        print("\nAnalyzing splits...")
        results = analyze_splits(splits_df, patient_mapping)
        
        # Print results
        print("\n" + "=" * 50)
        print("Split Validation Analysis")
        print("=" * 50)
        
        for split_name in ['train', 'val', 'test']:
            split_results = results[split_name]
            total = split_results['total']
            matched = split_results['matched']
            valid = split_results['valid']
            
            matched_pct = (matched / total * 100) if total > 0 else 0
            valid_pct = (valid / total * 100) if total > 0 else 0
            
            print(f"\n{split_name.capitalize()}:")
            print(f"  Total patients: {total}")
            print(f"  Matched patients: {matched} ({matched_pct:.1f}%)")
            print(f"  Patients with valid vital data: {valid} ({valid_pct:.1f}%)")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
