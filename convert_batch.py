#!/usr/bin/env python3
"""
Downsample image stacks in a directory using ImagingStack.

- Scans an input directory for .nii, .nii.gz, .nrrd, .nd2 files
- Loads each file with ImagingStack.from_file()
- Downsamples by specified factor in all spatial dimensions
- Saves as NIfTI: <stem>-01_ds.nii.gz into the output directory

Arguments:
- -i or --input: input directory
- -o or --output: output directory
- -d or --downsample: downsample factor

Usage:
    python downsample_with_imagingstack.py --input /path/to/input --output /path/to/output --downsample .5
"""

import os
import sys
import argparse
from typing import List

from ImagingStack import ImagingStack


SUPPORTED_EXTENSIONS: List[str] = [".nii", ".nii.gz", ".nrrd", ".nd2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample NIfTI/NRRD/ND2 files by 0.5 using ImagingStack."
    )
    parser.add_argument(
        "-d", "--downsample",
        required=True,
        help="Downsample factor to downasmple."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing image files."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for downsampled NIfTI files."
    )
    return parser.parse_args()


def is_supported(filename: str) -> bool:
    """Return True if filename has a supported extension."""
    fname = filename.lower()
    if fname.endswith(".nii.gz"):
        return True
    return any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS if ext != ".nii.gz")


def make_output_name(filename: str) -> str:
    """
    Given an input filename, construct '<stem>-01_ds.nii.gz'.
    Handles .nii, .nii.gz, .nrrd, .nd2.
    """
    fname = filename
    lower = fname.lower()

    if lower.endswith(".nii.gz"):
        stem = fname[:-7]  # remove '.nii.gz'
    elif lower.endswith(".nii"):
        stem = fname[:-4]  # remove '.nii'
    elif lower.endswith(".nrrd"):
        stem = fname[:-5]  # remove '.nrrd'
    elif lower.endswith(".nd2"):
        stem = fname[:-4]  # remove '.nd2'
    else:
        stem, _ = os.path.splitext(fname)

    return f"{stem}-01_ds.nii.gz"


def main() -> None:
    args = parse_args()

    input_dir = args.input
    output_dir = args.output
    downsample = args.downsample

    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist or is not a directory.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    for fname in files:
        if not is_supported(fname):
            continue

        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue

        try:
            # Load with your ImagingStack class
            stack = ImagingStack.from_file(in_path)

            # Downsample in x, y, z
            ds_stack = stack.downsample(downsample, downsample, downsample)

            out_name = make_output_name(fname)
            out_path = os.path.join(output_dir, out_name)

            # ImagingStack.write() should infer NIfTI from .nii.gz
            ds_stack.write(out_path)

            print(f"Processed: {fname} -> {out_name}")

        except Exception as e:
            print(f"# ERROR processing {fname}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()