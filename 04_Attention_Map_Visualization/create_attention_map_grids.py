#!/usr/bin/env python
"""
Create 3x3 grid figures from attention map images across 7 model folders.
Takes an AttentionMaps root directory, matches images by filename across subfolders,
arranges them in a 3x3 grid (PRUNED at bottom center, others in top 6 slots),
labels each with the folder name, and exports at 300 dpi.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


EXPECTED_FOLDER_NAMES = frozenset({
    "CONCH_V15", "NAIVE", "GIGAPATH", "HOPTIMUS1", "MUSK", "VIRCHOW2", "PRUNED",
})
PRUNED_NAME = "PRUNED"
# Non-PRUNED in reproducible order (alphabetical) for top 6 slots
NON_PRUNED_ORDER = ["CONCH_V15", "GIGAPATH", "HOPTIMUS1", "MUSK", "NAIVE", "VIRCHOW2"]
DEFAULT_PATTERNS = ["*_attention.png", "*_tumor_normal.png", "*_other_tissues.png"]


def discover_folders(input_dir: Path):
    """
    Return (list of folder paths in display order, pruned_path) or raise.
    Display order: 6 non-PRUNED then PRUNED.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
    names = {p.name.upper(): p for p in subdirs}
    # Normalize for comparison (allow PRUNED/Pruned)
    found = set()
    for n in EXPECTED_FOLDER_NAMES:
        if n in names:
            found.add(n)
    if len(found) != 7:
        got = sorted(names.keys())
        raise SystemExit(
            f"Expected exactly 7 subfolders among {sorted(EXPECTED_FOLDER_NAMES)}. "
            f"Found: {got}"
        )

    ordered = [names[n] for n in NON_PRUNED_ORDER]
    pruned_path = names[PRUNED_NAME]
    return ordered, pruned_path


def collect_by_basename(folder_paths, pruned_path, pattern: str):
    """
    For one glob pattern, return list of (basename, dict folder_name -> path).
    Only include basenames that appear in all 7 folders.
    """
    all_folders = folder_paths + [pruned_path]
    folder_names = [p.name for p in folder_paths] + [PRUNED_NAME]

    # per folder: basename -> path
    by_folder = {}
    for folder, name in zip(all_folders, folder_names):
        by_folder[name] = {f.name: f for f in folder.glob(pattern)}

    # basenames present in every folder
    first = folder_names[0]
    common_basenames = set(by_folder[first].keys())
    for name in folder_names[1:]:
        common_basenames &= set(by_folder[name].keys())

    out = []
    for basename in sorted(common_basenames):
        mapping = {name: by_folder[name][basename] for name in folder_names}
        out.append((basename, mapping))
    return out


def build_grid_for_basename(basename: str, folder_to_path: dict, output_path: Path, dpi: int = 300):
    """
    Create one 3x3 figure: slots 0-5 = non-PRUNED, 7 = PRUNED, 6 and 8 empty.
    """
    fig, axes = plt.subplots(3, 3)
    axes_flat = axes.flat

    # Slots 0-5 = non-PRUNED, 7 = PRUNED (bottom center), 6 and 8 = empty
    slot_for_folder = {name: i for i, name in enumerate(NON_PRUNED_ORDER)}
    slot_for_folder[PRUNED_NAME] = 7

    for folder_name, path in folder_to_path.items():
        slot = slot_for_folder[folder_name.upper()]
        ax = axes_flat[slot]
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.set_title(folder_name)
        ax.set_axis_off()

    for idx in (6, 8):
        axes_flat[idx].set_axis_off()
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create 3x3 grid figures from attention map images across 7 model folders."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Path to AttentionMaps root (directory containing the 7 subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write grid PNGs; default: {input_dir}/grids",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob for image files (e.g. *_attention.png). If omitted, processes "
             "*_attention.png, *_tumor_normal.png, *_other_tissues.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "grids"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_paths, pruned_path = discover_folders(input_dir)

    patterns = [args.pattern] if args.pattern else DEFAULT_PATTERNS
    total = 0
    for pattern in patterns:
        matched = collect_by_basename(folder_paths, pruned_path, pattern)
        for basename, folder_to_path in matched:
            stem = Path(basename).stem
            out_name = f"{stem}_grid.png"
            out_path = output_dir / out_name
            build_grid_for_basename(basename, folder_to_path, out_path, dpi=args.dpi)
            total += 1
            print(out_path)

    if total == 0:
        print("No matched image sets found.", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {total} grid(s) to {output_dir}")


if __name__ == "__main__":
    main()
