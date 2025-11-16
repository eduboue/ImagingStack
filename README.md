# ImagingStack

A lightweight, general-purpose Python class for loading, representing, downsampling, and saving multi-channel microscopy image stacks.  
Supports **NRRD**, **NIfTI (.nii / .nii.gz)**, and **ND2** microscopy files, and provides a unified interface to work with confocal, two-photon, or multi-channel volumetric imaging data.

---

## Features

### ✔ Load Multiple Imaging Formats

- **NRRD**
- **NIfTI** (`.nii`, `.nii.gz`)
- **ND2** (Nikon microscopy format)

### ✔ Multi-Channel Support

- Up to **three channels** stored as aligned `(Z, Y, X)` NumPy arrays.
- Automatic detection and splitting of channels from 4D volume arrays and ND2 files.

### ✔ Metadata Handling

- Stores and propagates metadata such as (will be specific to file type):
  - voxel size
  - `pixdim`
  - `space directions`
  - ND2 acquisition metadata (axes, raw metadata blob)

### ✔ Downsampling

Two methods:

- **Nearest-neighbor** (fast subsampling via evenly spaced indices).
- **Average pooling** (block-mean downsampling, like 3D binning).

Each operates independently on all channels and updates metadata accordingly where possible.

### ✔ Saving / Exporting

Save any channel as:

- **NRRD**
- **NIfTI**
- **Compressed NIfTI (.nii.gz)**

Preserves or converts metadata appropriately for each file type (e.g., voxel size, `space directions`, `pixdim`).

---

## Installation

You can install the dependencies into a virtual environment or conda environment. This was made with Python 12

```bash
pip install numpy nibabel pynrrd nd2
```

`ImagingStack.py` itself is just a single-file module; you can either:

- keep it in the same directory as your analysis scripts / notebooks, or  
- install it as a package if you later decide to structure this as a full Python package.

---

## Quick Start

### 1. Import and Load a Volume

```python
from ImagingStack import ImagingStack

# Load from NIfTI or NRRD
stack = ImagingStack.from_file("brain_volume.nii.gz")
print(stack)
```

### 2. Load From an ND2 Microscopy File

```python
stack = ImagingStack.from_nd2("experiment.nd2")
print(stack.channel1.shape)  # Z, Y, X
```

### 3. Downsample a Stack

```python
# Uniform 2x downsample in all dimensions using block averaging
small = stack.downsample(scale=0.5, method="average")

# Or use nearest-neighbor
smaller_fast = stack.downsample(scale=0.25, method="nearest")
```

### 4. Save a Channel to NIfTI or NRRD

```python
# Save channel 1 as NIfTI
small.to_file("output_downsampled.nii.gz", channel=1)

# Save channel 2 as NRRD
small.to_file("output_channel2.nrrd", channel=2)
```

For a more complete walkthrough—including inspecting metadata, visualizing slices, and comparing different downsampling methods—see the notebook:

> **`ImageStack_workflow.ipynb`** – step-by-step usage guide with runnable examples.

---

## API Overview

### `ImagingStack`

Core container holding up to three channels plus metadata.

- **Attributes**
  - `channel1`, `channel2`, `channel3` – NumPy arrays with shape `(Z, Y, X)`.
  - `metadata` – dictionary for voxel size, acquisition info, and file-format-specific fields.
  - `data` – property alias for `channel1` (for backward compatibility).

- **Class Methods**
  - `ImagingStack.from_file(filename)`  
    Load `.nrrd`, `.nii`, or `.nii.gz` and auto-detect/split channels.
  - `ImagingStack.from_nd2(filename, metadata=None)`  
    Load an ND2 file, extract axes and metadata, and split channels into `(Z, Y, X)` arrays.

- **Instance Methods**
  - `downsample(scale=0.5, method="nearest")`  
    Downsample all channels with either evenly spaced nearest-neighbor sampling or block-mean averaging.
  - `to_file(filename, fmt=None, channel=1)`  
    Save a single channel as NRRD or NIfTI, with metadata mapped to the appropriate header fields.

---

## Notes & Caveats

- Average downsampling assumes scale factors that correspond roughly to integer block sizes (e.g., 0.5 → block size 2, 0.25 → block size 4). Non-integer choices may lead to slightly unintuitive block sizes due to integer rounding.
- In average mode, edges may be trimmed so that each dimension is exactly divisible by the block size.
- Nearest-neighbor downsampling uses `np.linspace` to select evenly spaced indices along each axis.
- Metadata handling is conservative and designed to be “good enough” for most analysis workflows; if you rely on very strict header semantics, you may want to add tests or extensions specific to your pipeline.

---

## Author

Erik R. Duboué  
eduboue@fau.edu
eduboue@gmail.com
Florida Atlantic University – Neuroscience & Biology

If you use this in a paper, talk, or teaching material, a short acknowledgment is always appreciated.
