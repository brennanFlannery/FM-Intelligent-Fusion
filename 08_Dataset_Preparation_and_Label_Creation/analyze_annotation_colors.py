#!/usr/bin/env python3
"""
Analyze Line Colors in Prostate Pathology Annotations

This script parses XML annotation files and extracts all unique LineColor values,
converting them from Windows COLORREF format (BGR) to RGB for readability.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def colorref_to_rgb(colorref):
    """
    Convert Windows COLORREF format (BGR as 32-bit integer) to RGB tuple.
    
    COLORREF format: 0x00BBGGRR (Blue-Green-Red)
    
    Args:
        colorref: Integer color value in COLORREF format
        
    Returns:
        tuple: (R, G, B) values in range 0-255
    """
    # Extract BGR components
    blue = (colorref >> 16) & 0xFF
    green = (colorref >> 8) & 0xFF
    red = colorref & 0xFF
    
    return (red, green, blue)


def rgb_to_hex(rgb):
    """
    Convert RGB tuple to hex color code.
    
    Args:
        rgb: Tuple of (R, G, B) values
        
    Returns:
        str: Hex color code in #RRGGBB format
    """
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def parse_annotation_file(xml_path):
    """
    Parse an XML annotation file and extract LineColor values.
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        list: List of LineColor values found in the file
    """
    colors = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find all Annotation elements
        for annotation in root.findall('.//Annotation'):
            line_color = annotation.get('LineColor')
            if line_color:
                try:
                    color_value = int(line_color)
                    colors.append(color_value)
                except ValueError:
                    print(f"Warning: Invalid LineColor value '{line_color}' in {xml_path}", file=sys.stderr)
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading {xml_path}: {e}", file=sys.stderr)
    
    return colors


def analyze_annotations(annotations_dir):
    """
    Analyze all XML files in the annotations directory.
    
    Args:
        annotations_dir: Path to directory containing XML annotation files
        
    Returns:
        dict: Dictionary mapping color values to metadata
    """
    color_stats = defaultdict(lambda: {'count': 0, 'files': set()})
    total_files = 0
    total_annotations = 0
    
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    # Find all XML files
    xml_files = list(annotations_path.glob('*.xml'))
    
    if not xml_files:
        print(f"Warning: No XML files found in {annotations_dir}", file=sys.stderr)
        return color_stats, total_files, total_annotations
    
    print(f"Processing {len(xml_files)} XML files...", file=sys.stderr)
    
    for xml_file in xml_files:
        total_files += 1
        colors = parse_annotation_file(xml_file)
        
        for color_value in colors:
            total_annotations += 1
            color_stats[color_value]['count'] += 1
            color_stats[color_value]['files'].add(xml_file.name)
    
    return color_stats, total_files, total_annotations


def print_report(color_stats, total_files, total_annotations):
    """
    Print a formatted report of color analysis results.
    
    Args:
        color_stats: Dictionary mapping color values to metadata
        total_files: Total number of files processed
        total_annotations: Total number of annotations found
    """
    print("\n" + "="*80)
    print("ANNOTATION LINE COLOR ANALYSIS REPORT")
    print("="*80)
    print(f"\nSummary Statistics:")
    print(f"  Total files processed: {total_files}")
    print(f"  Total annotations found: {total_annotations}")
    print(f"  Unique colors found: {len(color_stats)}")
    
    if not color_stats:
        print("\nNo colors found in annotations.")
        return
    
    # Sort colors by count (descending)
    sorted_colors = sorted(color_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"\n{'='*80}")
    print("COLOR DETAILS")
    print(f"{'='*80}")
    print(f"{'Decimal':<12} {'RGB':<20} {'Hex':<10} {'Count':<8} {'Sample Files'}")
    print(f"{'-'*80}")
    
    for color_value, stats in sorted_colors:
        rgb = colorref_to_rgb(color_value)
        hex_code = rgb_to_hex(rgb)
        count = stats['count']
        
        # Show up to 3 sample files
        sample_files = list(stats['files'])[:3]
        files_str = ', '.join(sample_files)
        if len(stats['files']) > 3:
            files_str += f" ... (+{len(stats['files']) - 3} more)"
        
        print(f"{color_value:<12} {str(rgb):<20} {hex_code:<10} {count:<8} {files_str}")
    
    print(f"{'='*80}\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Analyze LineColor values in XML annotation files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/annotations
  %(prog)s /path/to/annotations --verbose
        """
    )
    
    parser.add_argument(
        'annotations_dir',
        nargs='?',
        default='/mnt/vstor/CSE_BME_CCIPD/home/bxf169/ScratchBackup/ProstatePathologyData/Annotations',
        help='Path to directory containing XML annotation files (default: %(default)s)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        color_stats, total_files, total_annotations = analyze_annotations(args.annotations_dir)
        print_report(color_stats, total_files, total_annotations)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
