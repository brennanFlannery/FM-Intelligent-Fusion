#!/usr/bin/env python
import argparse
import sys
import re
from pathlib import Path

import openslide
import h5py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Try to import shapely for polygon offsetting, fallback to simple method if unavailable
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# =============================================================================
# Utility functions
# =============================================================================

def infer_tiling_from_path(attn_dir: Path):
    """Infer tile_size (px) & extraction magnification (x) from folder name like '20x_512px'."""
    m = re.search(r"(?P<mag>\d+)x[\-_](?P<size>\d+)px", str(attn_dir))
    if m:
        return int(m.group("size")), int(m.group("mag"))
    return None, None


def load_attention(slide_stem: str, attention_dir: Path):
    """Load tile origins (coords) and attention values from a CLAM .h5 file."""
    h5_path = attention_dir / f"{slide_stem}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Attention file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        coords = np.array(f.get('coords', f.get('coord'))).reshape(-1, 2)
        attn   = np.array(f.get('attention', f.get('attn'))).reshape(-1)
    return coords, attn


def colorref_to_bgr(colorref):
    """Convert COLORREF (0x00BBGGRR) to BGR tuple for OpenCV.
    
    Args:
        colorref: Integer color value in COLORREF format, or None/string
        
    Returns:
        tuple: (B, G, R) values in range 0-255 for OpenCV
    """
    if colorref is None:
        return (0, 0, 255)  # Default to red
    try:
        colorref = int(colorref)
        b = (colorref >> 16) & 0xFF
        g = (colorref >> 8) & 0xFF
        r = colorref & 0xFF
        return (b, g, r)
    except (ValueError, TypeError):
        return (0, 0, 255)  # Default to red on error


# Prostate annotation color constants (COLORREF format)
PROSTATE_TUMOR_COLOR = 65280   # Green (#00FF00) - represents tumor in annotations
PROSTATE_BENIGN_COLOR = 65535  # Yellow (#FFFF00) - represents benign tissue in annotations


def parse_annotation_kidney(xml_path: Path):
    """Parse *.tumor.xml (kidney format); return list of all polygons (all regions are tumor)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    tumor_polys = []
    for region in tree.iter('Region'):
        pts = []
        verts = region.find('Vertices')
        if verts is None:
            continue
        for v in verts.iter('Vertex'):
            pts.append((float(v.get('X')), float(v.get('Y'))))
        if len(pts) >= 3:
            tumor_polys.append(np.array(pts))
    return tumor_polys


def parse_annotation_prostate(xml_path: Path):
    """Parse prostate format XML with color filtering; return (tumor_polys, benign_polys) tuple.
    
    Prostate format structure:
    <Annotations> -> <Annotation> -> <Regions> -> <Region> -> <Vertices> -> <Vertex>
    
    Filters annotations by LineColor:
    - Green (65280) = Tumor → returned as tumor_polys
    - Yellow (65535) = Benign → returned as benign_polys
    - Other colors = Ignored
    
    Returns:
        tuple: (tumor_polys, benign_polys) where each is a list of polygons
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tumor_polys = []
    benign_polys = []
    
    # Iterate through all Annotation elements
    for annotation in root.findall('Annotation'):
        # Extract LineColor from annotation
        line_color = annotation.get('LineColor')
        colorref = int(line_color) if line_color else None
        
        # Iterate through all Region elements within this annotation
        for region in annotation.findall('.//Region'):
            pts = []
            verts = region.find('Vertices')
            if verts is None:
                continue
            for v in verts.iter('Vertex'):
                x = v.get('X')
                y = v.get('Y')
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                poly = np.array(pts)
                # Filter by color
                if colorref == PROSTATE_TUMOR_COLOR:
                    tumor_polys.append(poly)
                elif colorref == PROSTATE_BENIGN_COLOR:
                    benign_polys.append(poly)
                # Other colors are ignored
    
    return tumor_polys, benign_polys


def parse_annotation_prostate_colored(xml_path: Path):
    """Parse prostate format XML with color support; return list of (polygon, bgr_color) tuples.
    
    Prostate format structure:
    <Annotations> -> <Annotation> -> <Regions> -> <Region> -> <Vertices> -> <Vertex>
    Each annotation's LineColor attribute is extracted and converted to BGR format.
    All regions within an annotation share the same color.
    
    Returns:
        list: List of (polygon, bgr_color) tuples where polygon is np.ndarray and
              bgr_color is (B, G, R) tuple for OpenCV
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation_polys = []
    
    # Iterate through all Annotation elements
    for annotation in root.findall('Annotation'):
        # Extract LineColor from annotation
        line_color = annotation.get('LineColor')
        bgr_color = colorref_to_bgr(line_color)
        
        # Iterate through all Region elements within this annotation
        for region in annotation.findall('.//Region'):
            pts = []
            verts = region.find('Vertices')
            if verts is None:
                continue
            for v in verts.iter('Vertex'):
                x = v.get('X')
                y = v.get('Y')
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                annotation_polys.append((np.array(pts), bgr_color))
    
    return annotation_polys


def find_annotation_xml(annotation_dir: Path, stem: str, data_type: str):
    """Return the annotation XML file path for given stem, or None.
    
    Args:
        annotation_dir: Directory containing annotation files
        stem: Slide filename stem (without extension)
        data_type: 'kidney', 'prostate', 'prostate_colored', 'rectal', or 'rectal_sra'
    
    Returns:
        Path to annotation file or None if not found
    """
    if data_type == 'kidney':
        # Kidney format: look for files matching {stem}*tumor*.xml
        files = list(annotation_dir.glob(f"{stem}*tumor*.xml"))
        return files[0] if files else None
    elif data_type in ('prostate', 'prostate_colored'):
        # Prostate format: look for exact match {stem}.xml
        xml_path = annotation_dir / f"{stem}.xml"
        return xml_path if xml_path.exists() else None
    elif data_type == 'rectal':
        # Rectal format: look in subfolder {stem}/{stem}.tumor.xml
        xml_path = annotation_dir / stem / f"{stem}.tumor.xml"
        return xml_path if xml_path.exists() else None
    elif data_type == 'rectal_sra':
        # Rectal SRA format: no single tumor XML, uses per-class XMLs
        return None
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def find_normal_xml(annotation_dir: Path, stem: str, data_type: str):
    """Return the normal annotation XML file path for given stem, or None.
    
    Args:
        annotation_dir: Directory containing annotation files
        stem: Slide filename stem (without extension)
        data_type: 'kidney', 'prostate', or 'rectal'
    
    Returns:
        Path to annotation file or None if not found
    """
    if data_type == 'kidney':
        # Kidney format: look for files matching {stem}*normal*.xml
        files = list(annotation_dir.glob(f"{stem}*normal*.xml"))
        return files[0] if files else None
    elif data_type == 'rectal':
        # Rectal format: look in subfolder {stem}/{stem}.normal.xml
        xml_path = annotation_dir / stem / f"{stem}.normal.xml"
        return xml_path if xml_path.exists() else None
    else:
        # Prostate format doesn't have separate normal annotations
        return None


# SRA tissue classes (output by SRA model; BACK omitted)
SRA_CLASSES = ['ADI', 'CSTR', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def find_sra_class_xml(annotation_dir: Path, stem: str, class_label: str):
    """Return the SRA class XML path for rectal format (subfolder per slide), or None.
    
    SRA annotations are stored as: annotation_dir/{stem}/{stem}.{class_label}.xml
    """
    xml_path = annotation_dir / stem / f"{stem}.{class_label}.xml"
    return xml_path if xml_path.exists() else None


def has_any_sra_annotation(annotation_dir: Path, stem: str) -> bool:
    """Return True if at least one SRA class XML exists for this slide under annotation_dir/{stem}/."""
    for cls in SRA_CLASSES:
        if find_sra_class_xml(annotation_dir, stem, cls) is not None:
            return True
    return False


def find_sra_npy_file(annotation_dir: Path, stem: str):
    """Return the SRA classification NPY file path for rectal format (subfolder per slide), or None.
    
    SRA NPY files are stored as: annotation_dir/{stem}/{stem}.svs_classification_sra.npy
    """
    npy_path = annotation_dir / stem / f"{stem}.svs_classification_sra.npy"
    return npy_path if npy_path.exists() else None


def load_sra_classification_npy(npy_path: Path) -> dict:
    """Load and validate SRA classification NPY file.
    
    Args:
        npy_path: Path to the NPY file
        
    Returns:
        Dictionary containing classification data with keys:
        - 'classification': numpy array of shape (N, n_classes) with class scores
        - 'metadata': numpy array of shape (N, 10) with patch coordinates and metadata
        - 'classification_labels': list of class label strings
        
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys are missing
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    
    data = np.load(npy_path, allow_pickle=True).item()
    
    # Validate required keys
    required_keys = ['classification', 'metadata', 'classification_labels']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in NPY file: {npy_path}")
    
    return data


def compute_global_majority_vote(npy_data: dict, sx: float, sy: float, H_thumb: int, W_thumb: int,
                                  exclude_classes: list = ['BACK', 'DEB']):
    """Compute global majority vote across all classes (except excluded ones).
    
    Args:
        npy_data: Dictionary from load_sra_classification_npy() containing:
            - 'classification': (N, n_classes) array of class scores
            - 'metadata': (N, 10) array with columns [mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar]
            - 'classification_labels': list of class label strings
        sx: Scale factor from full-res to thumbnail (x-axis)
        sy: Scale factor from full-res to thumbnail (y-axis)
        H_thumb: Thumbnail height in pixels
        W_thumb: Thumbnail width in pixels
        exclude_classes: List of class labels to exclude from voting
        
    Returns:
        tuple: (winner_map, class_idx_to_label, label_to_class_idx)
            - winner_map: (H_thumb, W_thumb) array with class index for each pixel (-1 if no votes)
            - class_idx_to_label: dict mapping class index to label string
            - label_to_class_idx: dict mapping label string to class index
    """
    classification = npy_data['classification']
    metadata = npy_data['metadata']
    classification_labels = npy_data['classification_labels']
    
    if classification.size == 0 or metadata.size == 0:
        return None, {}, {}
    
    # Get predicted class for each patch (argmax of classification scores)
    predicted_classes = np.argmax(classification, axis=1)
    
    # Extract patch coordinates from metadata
    tx = metadata[:, 2]  # top-left x
    ty = metadata[:, 3]  # top-left y
    bx = metadata[:, 6]  # bottom-right x
    by = metadata[:, 7]  # bottom-right y
    
    # Build mapping: only include classes that are in SRA_CLASSES and not excluded
    class_idx_to_label = {}
    label_to_class_idx = {}
    active_class_indices = []
    
    for cls_idx, cls_label in enumerate(classification_labels):
        if cls_label not in SRA_CLASSES:
            continue
        if cls_label in exclude_classes:
            continue
        
        class_idx_to_label[cls_idx] = cls_label
        label_to_class_idx[cls_label] = cls_idx
        active_class_indices.append(cls_idx)
    
    if not active_class_indices:
        return None, {}, {}
    
    n_active = len(active_class_indices)
    idx_to_local = {cls_idx: local_idx for local_idx, cls_idx in enumerate(active_class_indices)}
    
    # Build votes array
    votes = np.zeros((n_active, H_thumb, W_thumb), dtype=np.uint16)
    
    for i in range(len(predicted_classes)):
        class_idx = int(predicted_classes[i])
        if class_idx not in idx_to_local:
            continue
        local_idx = idx_to_local[class_idx]
        
        # Scale coordinates from full-res to thumbnail
        tx_t = max(0, int(np.floor(tx[i] * sx)))
        ty_t = max(0, int(np.floor(ty[i] * sy)))
        bx_t = min(W_thumb, int(np.ceil(bx[i] * sx)))
        by_t = min(H_thumb, int(np.ceil(by[i] * sy)))
        
        if bx_t > tx_t and by_t > ty_t:
            votes[local_idx, ty_t:by_t, tx_t:bx_t] += 1
    
    # Winner-takes-all
    max_votes = votes.max(axis=0)
    winner_local = votes.argmax(axis=0)
    has_votes = max_votes > 0
    
    # Convert local indices back to class indices
    winner_map = np.full((H_thumb, W_thumb), -1, dtype=np.int32)
    for local_idx, class_idx in enumerate(active_class_indices):
        winner_map[(winner_local == local_idx) & has_votes] = class_idx
    
    return winner_map, class_idx_to_label, label_to_class_idx


def draw_from_winner_map(overlay: np.ndarray, winner_map: np.ndarray, 
                         class_idx_to_label: dict, include_classes: list,
                         dataset_name: str = 'kather19crctp', 
                         line_thickness: int = 20,
                         outline_offset: float = 0.0) -> bool:
    """Draw outlines from a pre-computed winner map, filtered by class list.
    
    Args:
        overlay: BGR image array to draw on (modified in place)
        winner_map: (H, W) array with class index for each pixel (-1 if no votes)
        class_idx_to_label: dict mapping class index to label string
        include_classes: List of class labels to draw
        dataset_name: Dataset name for color mapping
        line_thickness: Thickness of border lines in pixels
        outline_offset: Inward erosion in thumbnail pixels before finding contours (default: 0.0)
        
    Returns:
        True if any regions were drawn, False otherwise
    """
    if winner_map is None or winner_map.size == 0:
        return False
    
    H_thumb, W_thumb = winner_map.shape
    
    # Filter: only draw classes in include_classes
    class_indices_to_draw = []
    for cls_idx, cls_label in class_idx_to_label.items():
        if cls_label in include_classes:
            class_indices_to_draw.append(cls_idx)
    
    if not class_indices_to_draw:
        return False
    
    regions_drawn = 0
    
    # Draw each class separately
    for class_idx in class_indices_to_draw:
        cls_label = class_idx_to_label[class_idx]
        
        # Create mask for this class
        class_mask = (winner_map == class_idx).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        
        color_rgb = get_sra_class_color(cls_label, dataset_name)
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # RGB to BGR
        
        # Erode mask inward before finding contours to avoid overlapping outlines
        if outline_offset > 0:
            kernel_size = max(1, int(round(outline_offset)))
            kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)
            class_mask = cv2.erode(class_mask, kernel, iterations=1)
            if class_mask.sum() == 0:
                continue

        # Find connected components
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw outlines for each connected region
        for contour in contours:
            if len(contour) < 4:
                continue
            cv2.polylines(overlay, [contour], True, color_bgr, line_thickness)
            regions_drawn += 1
    
    return regions_drawn > 0


def draw_patch_borders_from_npy(overlay: np.ndarray, npy_data: dict, sx: float, sy: float, 
                                 dataset_name: str = 'kather19crctp', 
                                 exclude_classes: list = ['BACK'],
                                 include_classes: list = None,
                                 line_thickness: int = 7) -> bool:
    """Draw patch borders on overlay image based on SRA classification from NPY data.
    
    Uses pixel-level majority voting to resolve overlapping patches: each pixel is assigned
    to the class with the most votes from overlapping patches. Connected regions of the same
    class are then merged into single regions with one outline per connected component.
    Uses 4-connectedness (edge-adjacent patches only). This ensures no overlapping outlines
    between different classes.
    
    Args:
        overlay: BGR image array to draw on (modified in place)
        npy_data: Dictionary from load_sra_classification_npy() containing:
            - 'classification': (N, n_classes) array of class scores
            - 'metadata': (N, 10) array with columns [mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar]
            - 'classification_labels': list of class label strings
        sx: Scale factor from full-res to thumbnail (x-axis)
        sy: Scale factor from full-res to thumbnail (y-axis)
        dataset_name: Dataset name for color mapping ('kather19' or 'kather19crctp')
        exclude_classes: List of class labels to exclude from drawing (used when include_classes is None)
        include_classes: Optional whitelist of class labels to draw. When specified, only patches
            whose class is in this list are drawn, and exclude_classes is ignored.
        line_thickness: Thickness of border lines in pixels
        
    Returns:
        True if any patches were drawn, False otherwise
    """
    classification = npy_data['classification']
    metadata = npy_data['metadata']
    classification_labels = npy_data['classification_labels']
    
    if classification.size == 0 or metadata.size == 0:
        return False
    
    # Get overlay dimensions
    H_thumb, W_thumb = overlay.shape[:2]
    
    # Get predicted class for each patch (argmax of classification scores)
    predicted_classes = np.argmax(classification, axis=1)
    
    # Create mapping from class index to label
    idx_to_label = {i: label for i, label in enumerate(classification_labels)}
    
    # Extract patch coordinates from metadata
    # metadata columns: [mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar]
    tx = metadata[:, 2]  # top-left x
    ty = metadata[:, 3]  # top-left y
    bx = metadata[:, 6]  # bottom-right x
    by = metadata[:, 7]  # bottom-right y
    
    # --- Step 1: Map include_classes to local indices ---
    # Build a mapping from classification_labels index → local vote index
    # Only track classes that will actually be drawn
    local_class_labels = []   # ordered list of active class labels
    label_to_local = {}       # classification_labels index → local vote index
    
    for cls_idx, cls_label in enumerate(classification_labels):
        # Skip if class not in SRA_CLASSES (only draw valid SRA classes)
        if cls_label not in SRA_CLASSES:
            continue
        
        # Apply whitelist (include_classes) or blacklist (exclude_classes) filter
        if include_classes is not None:
            if cls_label not in include_classes:
                continue
        else:
            if cls_label in exclude_classes:
                continue
        
        label_to_local[cls_idx] = len(local_class_labels)
        local_class_labels.append(cls_label)
    
    if not local_class_labels:
        return False
    
    n_local = len(local_class_labels)
    
    # --- Step 2: Build votes array ---
    votes = np.zeros((n_local, H_thumb, W_thumb), dtype=np.uint16)
    
    for i in range(len(predicted_classes)):
        class_idx = int(predicted_classes[i])
        if class_idx not in label_to_local:
            continue
        local_idx = label_to_local[class_idx]
        
        # Scale coordinates from full-res to thumbnail
        # Use floor for top-left, ceil for bottom-right to avoid gaps
        tx_t = max(0, int(np.floor(tx[i] * sx)))
        ty_t = max(0, int(np.floor(ty[i] * sy)))
        bx_t = min(W_thumb, int(np.ceil(bx[i] * sx)))
        by_t = min(H_thumb, int(np.ceil(by[i] * sy)))
        
        if bx_t > tx_t and by_t > ty_t:
            votes[local_idx, ty_t:by_t, tx_t:bx_t] += 1
    
    # --- Step 3: Winner-takes-all ---
    max_votes = votes.max(axis=0)
    winner = votes.argmax(axis=0)
    has_votes = max_votes > 0
    
    # --- Step 4: Per-class masks → contours → draw ---
    regions_drawn = 0
    for local_idx, cls_label in enumerate(local_class_labels):
        class_mask = ((winner == local_idx) & has_votes).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        
        color_rgb = get_sra_class_color(cls_label, dataset_name)
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # RGB to BGR
        
        # Find connected components (4-connected, edge-adjacent only)
        # RETR_EXTERNAL retrieves only the outer contours (no holes)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw outlines for each connected region
        for contour in contours:
            # Skip very small contours (less than 4 points can't form a valid polygon)
            if len(contour) < 4:
                continue
            
            # Draw polygon outline
            cv2.polylines(overlay, [contour], True, color_bgr, line_thickness)
            regions_drawn += 1
    
    return regions_drawn > 0


def get_sra_class_color(class_label: str, dataset_name: str = 'kather19crctp'):
    """Get RGB color for an SRA class label based on dataset colormap.
    
    Args:
        class_label: SRA class label (e.g., 'ADI', 'TUM', 'CSTR')
        dataset_name: Dataset name ('kather19' or 'kather19crctp')
    
    Returns:
        tuple: (R, G, B) values in range 0-1 (normalized)
    """
    # Color mapping based on build_disrete_cmap from SRA repo
    if dataset_name == 'kather19':
        class_colors = {
            'ADI': [247, 129, 191],  # Pink
            'BACK': [153, 153, 153],  # Gray
            'DEB': [255, 255, 51],  # Yellow
            'LYM': [255, 0, 255],  # Magenta
            'MUC': [23, 190, 192],  # Cyan
            'MUS': [255, 127, 0],  # Orange
            'NORM': [0, 255, 0],  # Bright Green
            'STR': [55, 126, 184],  # Blue
            'TUM': [228, 26, 28],  # Red
        }
    elif dataset_name == 'kather19crctp':
        class_colors = {
            'ADI': [247, 129, 191],  # Pink
            'BACK': [153, 153, 153],  # Gray
            'DEB': [255, 255, 51],  # Yellow
            'LYM': [255, 0, 255],  # Magenta
            'MUC': [23, 190, 192],  # Cyan
            'MUS': [255, 127, 0],  # Orange
            'NORM': [0, 255, 0],  # Bright Green
            'STR': [55, 126, 184],  # Blue
            'TUM': [228, 26, 28],  # Red
            'CSTR': [0, 128, 0],  # Dark Green
        }
    else:
        # Fallback: use a default color scheme
        class_colors = {
            'ADI': [247, 129, 191],
            'CSTR': [0, 128, 0],  # Dark Green
            'DEB': [255, 255, 51],
            'LYM': [255, 0, 255],  # Magenta
            'MUC': [23, 190, 192],
            'MUS': [255, 127, 0],
            'STR': [55, 126, 184],
            'TUM': [228, 26, 28],
        }
    
    # Get color and normalize to 0-1 range
    color = class_colors.get(class_label, [128, 128, 128])  # Default gray if unknown
    return tuple(c / 255.0 for c in color)


def offset_polygon_inward(polygon: np.ndarray, offset_pixels: float, sx: float, sy: float) -> np.ndarray:
    """Offset polygon vertices inward by a fixed pixel amount at thumbnail scale.
    
    Strategy:
    1. Convert polygon to Shapely Polygon if available
    2. Use buffer with negative distance to shrink inward
    3. Extract coordinates back
    4. Fallback to simple edge-based offsetting if Shapely unavailable
    
    Args:
        polygon: Nx2 array of polygon vertices (full-res coordinates)
        offset_pixels: Offset distance in thumbnail pixels
        sx, sy: Scale factors from full-res to thumbnail
    
    Returns:
        Offset polygon as Nx2 array (full-res coordinates)
    """
    if len(polygon) < 3:
        return polygon  # Cannot offset degenerate polygons
    
    # Convert offset from thumbnail pixels to full-res coordinates
    # Use average scale factor for simplicity
    avg_scale = (sx + sy) / 2.0
    offset_full_res = offset_pixels / avg_scale
    
    # Check if polygon is too small to offset
    # Compute approximate area
    area = 0.0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]
    area = abs(area) / 2.0
    
    # If area is too small, skip offsetting
    min_area = (2 * offset_full_res) ** 2
    if area < min_area:
        return polygon
    
    if SHAPELY_AVAILABLE:
        try:
            # Convert to Shapely polygon
            shapely_poly = ShapelyPolygon(polygon)
            
            # Buffer inward (negative distance shrinks)
            buffered = shapely_poly.buffer(-offset_full_res, join_style=2)  # join_style=2 is mitre
            
            # Handle case where buffer results in empty or invalid geometry
            if buffered.is_empty or not buffered.is_valid:
                return polygon
            
            # Extract coordinates
            if hasattr(buffered, 'exterior'):
                coords = np.array(buffered.exterior.coords[:-1])  # Remove duplicate last point
            else:
                # MultiPolygon or other - take first polygon
                if hasattr(buffered, 'geoms'):
                    coords = np.array(buffered.geoms[0].exterior.coords[:-1])
                else:
                    return polygon
            
            if len(coords) >= 3:
                return coords
            else:
                return polygon
        except Exception:
            # Fallback to simple method if Shapely fails
            pass
    
    # Fallback: Simple edge-based offsetting
    # Compute inward normal for each edge and shift vertices
    n = len(polygon)
    offset_poly = np.zeros_like(polygon)
    
    for i in range(n):
        # Previous and next vertices
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        
        # Edge vectors
        edge1 = polygon[i] - polygon[prev_idx]
        edge2 = polygon[next_idx] - polygon[i]
        
        # Normalize
        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            offset_poly[i] = polygon[i]
            continue
        
        edge1_norm = edge1 / len1
        edge2_norm = edge2 / len2
        
        # Compute angle bisector
        bisector = edge1_norm + edge2_norm
        bisector_len = np.linalg.norm(bisector)
        
        if bisector_len < 1e-6:
            # Parallel edges - use perpendicular to edge
            perp = np.array([-edge1[1], edge1[0]]) / len1
            offset_poly[i] = polygon[i] - perp * offset_full_res
        else:
            bisector_norm = bisector / bisector_len
            
            # Determine inward direction
            # The bisector of two edges at a vertex typically points outward
            # For inward offset, we need to move opposite to the bisector direction
            # Compute which side is interior using cross product
            # For CCW polygon: cross > 0, interior is to left, bisector points right (outward)
            # For CW polygon: cross < 0, interior is to right, bisector points left (outward)
            # In both cases, inward is opposite to bisector direction
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            # Always negate bisector for inward direction
            bisector_norm = -bisector_norm
            
            # Offset along bisector
            # Adjust offset distance based on angle
            angle = np.arccos(np.clip(np.dot(edge1_norm, edge2_norm), -1, 1))
            if angle < 1e-6:
                offset_dist = offset_full_res
            else:
                offset_dist = offset_full_res / np.sin(angle / 2)
            
            offset_poly[i] = polygon[i] + bisector_norm * offset_dist
    
    return offset_poly


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps on WSIs with tumor/benign annotations."
    )
    parser.add_argument('--slide_dir',      required=True, help='Directory with .svs slides')
    parser.add_argument('--attention_dir',  required=True, help='Directory with CLAM .h5 attention files')
    parser.add_argument('--annotation_dir', required=True, help='Directory with annotation XML files')
    parser.add_argument('--output_dir',     required=True, help='Directory to save PNGs')
    parser.add_argument('--data_type',      type=str, default='kidney', choices=['kidney', 'prostate', 'prostate_colored', 'rectal', 'rectal_sra'],
                       help='Dataset type: "kidney" (default), "prostate", "prostate_colored" (color-aware), "rectal", or "rectal_sra" (SRA subclasses)')
    parser.add_argument('--num_slides',     type=int, default=10,   help='Number of slides to process (-1 for all)')
    parser.add_argument('--target_dim',     type=int, default=4000, help='Longest side of thumbnail (px)')
    parser.add_argument('--tile_size',      type=int, required=True, help='Tile size at extraction mag (px)')
    parser.add_argument('--blur_sigma',     type=float, default=1.0, help='Gaussian blur sigma (thumbnail units)')
    parser.add_argument('--outline_offset', type=float, default=3.0, help='Inward offset for polygon outlines in thumbnail pixels (for rectal_sra, default: 3.0)')
    parser.add_argument('--sra_dataset',    type=str, default='kather19crctp', choices=['kather19', 'kather19crctp'],
                       help='SRA dataset name for color mapping (default: kather19crctp)')
    args = parser.parse_args()

    slide_dir = Path(args.slide_dir)
    attn_dir  = Path(args.attention_dir)
    ann_dir   = Path(args.annotation_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer extraction magnification
    _, extraction_mag = infer_tiling_from_path(attn_dir)
    if extraction_mag is None:
        sys.exit('Cannot infer extraction magnification from attention_dir path.')

    slides = sorted(slide_dir.glob('*.svs'))
    if not slides:
        sys.exit('No SVS files found')

    valids = [s for s in slides if (attn_dir / f"{s.stem}.h5").exists()]
    if not valids:
        sys.exit('No slides with attention files')
    
    # For rectal_sra, filter to only slides with SRA NPY files
    if args.data_type == 'rectal_sra':
        valids = [s for s in valids if find_sra_npy_file(ann_dir, s.stem) is not None]
        if not valids:
            sys.exit('No slides with both attention files and SRA classification NPY files found')

    # Process all slides if num_slides is -1
    slides_to_process = valids if args.num_slides == -1 else valids[:args.num_slides]
    
    for slide_path in tqdm(slides_to_process, desc='Processing slides'):
        # Open slide and get dimensions
        slide = openslide.OpenSlide(str(slide_path))
        W_full, H_full = slide.dimensions
        base_mag = float(slide.properties.get('openslide.objective-power', extraction_mag))
        mag_ratio = base_mag / extraction_mag

        # Generate thumbnail
        scale = args.target_dim / max(W_full, H_full)
        thumb = slide.get_thumbnail((int(W_full * scale), int(H_full * scale)))
        thumb_arr = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGBA2RGB)
        slide.close()

        H_thumb, W_thumb = thumb_arr.shape[:2]
        sx = W_thumb / W_full
        sy = H_thumb / H_full

        # Load and normalize attention
        coords, attn = load_attention(slide_path.stem, attn_dir)
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)

        # Compute tile size in thumbnail coordinates
        tile_full = args.tile_size * mag_ratio
        tw = int(np.floor(tile_full * sx))
        th = int(np.floor(tile_full * sy))

        # Rasterise attention
        heat = np.zeros((H_thumb, W_thumb), dtype=np.float32)
        for (x_f, y_f), a in zip(coords, attn_norm):
            x0 = int(np.floor(x_f * sx))
            y0 = int(np.floor(y_f * sy))
            x1 = min(W_thumb, x0 + tw)
            y1 = min(H_thumb, y0 + th)
            heat[y0:y1, x0:x1] = a

                # Apply smoothing
        heat = gaussian_filter(heat, sigma=args.blur_sigma)
        heat = np.clip(heat, 0, 1)

        # Threshold: zero out values below the 50th percentile
        nonzero = heat[heat > 0]
        if nonzero.size > 0:
            cutoff = np.percentile(nonzero, 50)
            heat[heat < cutoff] = 0


                # Composite non-zero heat regions with translucency
        heat_uint = (heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)
        overlay = thumb_arr.copy()
        mask = heat_uint > 0
        alpha = 0.3  # translucency level for heatmap
        blended = cv2.addWeighted(thumb_arr, 1 - alpha, heat_color, alpha, 0)
        overlay[mask] = blended[mask]

                # Draw tumor annotations on top
        has_tumor = False
        
        # Handle rectal_sra separately (uses patch-level NPY classifications)
        if args.data_type == 'rectal_sra':
            # Rectal SRA format: load patch-level classifications from NPY file
            npy_path = find_sra_npy_file(ann_dir, slide_path.stem)
            if npy_path is not None:
                try:
                    npy_data = load_sra_classification_npy(npy_path)

                    # Compute global majority vote once (excludes BACK and DEB)
                    winner_map, class_idx_to_label, label_to_class_idx = compute_global_majority_vote(
                        npy_data, sx, sy, H_thumb, W_thumb, exclude_classes=['BACK', 'DEB']
                    )
                    
                    if winner_map is None:
                        has_tumor = False
                    else:
                        # --- Overlay 1: Tumor + Normal ---
                        overlay_tn = overlay.copy()
                        has_tn = draw_from_winner_map(
                            overlay_tn, winner_map, class_idx_to_label,
                            include_classes=['TUM', 'NORM'],
                            dataset_name=args.sra_dataset,
                            line_thickness=15,
                            outline_offset=7,
                        )
                        if has_tn:
                            out_path_tn = out_dir / f"{slide_path.stem}_tumor_normal.png"
                            cv2.imwrite(str(out_path_tn), overlay_tn)

                        # --- Overlay 2: Other tissues (ADI, CSTR, LYM, MUC, MUS, STR) ---
                        overlay_ot = overlay.copy()
                        has_ot = draw_from_winner_map(
                            overlay_ot, winner_map, class_idx_to_label,
                            include_classes=['ADI', 'CSTR', 'LYM', 'MUC', 'MUS', 'STR'],
                            dataset_name=args.sra_dataset,
                            line_thickness=15,
                            outline_offset=7,
                        )
                        if has_ot:
                            out_path_ot = out_dir / f"{slide_path.stem}_other_tissues.png"
                            cv2.imwrite(str(out_path_ot), overlay_ot)

                        has_tumor = has_tn or has_ot

                except (FileNotFoundError, KeyError) as e:
                    # Skip this slide if NPY file is invalid
                    print(f"Warning: Failed to load NPY file for {slide_path.stem}: {e}")
                    has_tumor = False
            else:
                has_tumor = False
        else:
            # Other data types: use single annotation XML
            xml = find_annotation_xml(ann_dir, slide_path.stem, args.data_type)
            if xml:
                # Use appropriate parser based on data type
                if args.data_type == 'kidney':
                    tumor_polys = parse_annotation_kidney(xml)
                    if tumor_polys:  # Check if any polygons were parsed
                        has_tumor = True
                        for poly in tumor_polys:
                            pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                            # red color for tumor
                            cv2.polylines(overlay, [pts_thumb], True, (0,0,255), 15)
                elif args.data_type == 'prostate_colored':
                    # Color-aware prostate format
                    annotation_polys = parse_annotation_prostate_colored(xml)
                    if annotation_polys:  # Check if any polygons were parsed
                        has_tumor = True
                        for poly, color in annotation_polys:
                            pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                            # Use color from annotation
                            cv2.polylines(overlay, [pts_thumb], True, color, 15)
                elif args.data_type == 'prostate':
                    # Prostate format: filter by color, draw tumor in red and benign in green
                    tumor_polys, benign_polys = parse_annotation_prostate(xml)
                    if tumor_polys:  # Check if any tumor polygons were parsed
                        has_tumor = True
                        for poly in tumor_polys:
                            pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                            # red color for tumor
                            cv2.polylines(overlay, [pts_thumb], True, (0,0,255), 15)
                    if benign_polys:  # Check if any benign polygons were parsed
                        has_tumor = True  # Mark as having annotations
                        for poly in benign_polys:
                            pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                            # green color for benign tissue
                            cv2.polylines(overlay, [pts_thumb], True, (0,255,0), 15)
                elif args.data_type == 'rectal':
                    # Rectal format: all red (unchanged behavior)
                    tumor_polys = parse_annotation_prostate(xml)
                    # For rectal, parse_annotation_prostate returns tuple, but we need single list
                    # So we'll combine tumor and benign as tumor for rectal (backward compatibility)
                    if isinstance(tumor_polys, tuple):
                        tumor_polys = tumor_polys[0] + tumor_polys[1]
                    if tumor_polys:  # Check if any polygons were parsed
                        has_tumor = True
                        for poly in tumor_polys:
                            pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                            # red color for tumor
                            cv2.polylines(overlay, [pts_thumb], True, (0,0,255), 15)
                else:
                    raise ValueError(f"Unknown data_type: {args.data_type}")

        # Check for normal annotations (supported for kidney and rectal formats, not rectal_sra)
        has_normal = False
        if args.data_type != 'rectal_sra':
            normal_xml = find_normal_xml(ann_dir, slide_path.stem, args.data_type)
            if normal_xml:
                # Use appropriate parser based on data type
                if args.data_type == 'kidney':
                    normal_polys = parse_annotation_kidney(normal_xml)
                elif args.data_type == 'rectal':
                    normal_polys_result = parse_annotation_prostate(normal_xml)
                    # For rectal, parse_annotation_prostate returns tuple, combine as single list
                    if isinstance(normal_polys_result, tuple):
                        normal_polys = normal_polys_result[0] + normal_polys_result[1]
                    else:
                        normal_polys = normal_polys_result
                else:
                    normal_polys = []
                
                if normal_polys:
                    has_normal = True
                    for poly in normal_polys:
                        pts_thumb = np.floor(np.stack([poly[:,0]*sx, poly[:,1]*sy], axis=1)).astype(int)
                        # green color for normal
                        cv2.polylines(overlay, [pts_thumb], True, (0,255,0), 15)

        # Only save PNG if there are annotations (tumor, normal, or SRA classes)
        # rectal_sra saves two separate overlays inline above; skip the generic save for it
        if args.data_type != 'rectal_sra' and (has_tumor or has_normal):
            out_path = out_dir / f"{slide_path.stem}_attention.png"
            cv2.imwrite(str(out_path), overlay)

    print('Done')

if __name__ == '__main__':
    main()
