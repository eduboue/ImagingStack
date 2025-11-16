import os
import numpy as np
import nibabel as nib
import nrrd
from typing import Any, Dict, Optional, Tuple, Union 


class ImagingStack:
    """
    A general container for confocal or two-photon imaging stacks.
    Loads NRRD or NIfTI files and stores:
      - data: np.ndarray
      - metadata: dict
    """

    def __init__(self, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        data : np.ndarray
            Raw imaging volume (z, y, x[, channels ...]).
        metadata : dict, optional
            Arbitrary metadata describing voxel size, acquisition info, etc.
        """
        self.data: np.ndarray = data
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    @classmethod
    def from_file(cls, filename: str):
        """
        Detect whether the file is NRRD or NIfTI (.nii or .nii.gz)
        and load it into an ImagingStack instance.

        Parameters
        ----------
        filename : str
            Path to the imaging file.

        Returns
        -------
        ImagingStack
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File does not exist: {filename}")

        # Determine file extension
        fname = filename.lower()

        if fname.endswith(".nrrd"):
            data, header = nrrd.read(filename)
            metadata = dict(header)
        elif fname.endswith(".nii") or fname.endswith(".nii.gz"):
            img = nib.load(filename)
            data = img.get_fdata(dtype=np.float32)  # Convert to clean array
            metadata = dict(img.header)
        else:
            raise ValueError(
                "Unsupported file type. Must be .nrrd, .nii, or .nii.gz"
            )

        # Return class instance
        return cls(data=data, metadata=metadata)

    def __repr__(self):
        """Return a concise string summary describing shape and metadata keys."""
        return (f"ImagingStack(data_shape={self.data.shape}, "
                f"metadata_keys={list(self.metadata.keys())})")
    
    def downsample(
        self, 
        scale: Union[float, Tuple[float, float, float]] = 0.5,
        method: str = "nearest"
        ) -> "ImagingStack":

        """
        Downsample the image volume by a scale factor.
        
        Parameters
        ----------
        scale : float or (float, float, float)
            Scaling factor(s) for (z, y, x). 
            Example: 0.5 makes the volume half-size in each dimension.
        method : str
            "nearest" or "average". 
            "nearest" = fast, uses slicing.
            "average" = smoother but slower (block averaging).

        Returns
        -------
        ImagingStack
            A new ImagingStack with downsampled data and updated metadata.
        """

        # Get the downsample factors and downsample using method nearest or average.
        # Normalize scale into a 3-tuple
        if isinstance(scale, (int, float)):
            sz = sy = sx = float(scale)
        else:
            if len(scale) != 3:
                raise ValueError("scale must be float or 3-tuple (z, y, x)")
            sz, sy, sx = map(float, scale)

        z, y, x = self.data.shape[:3]

        # NEW dimensions
        new_z = max(1, int(z * sz))
        new_y = max(1, int(y * sy))
        new_x = max(1, int(x * sx))

        if method == "nearest":
            # Simple subsampling
            ds = self.data[
                np.linspace(0, z - 1, new_z).astype(int)[:, None, None],
                np.linspace(0, y - 1, new_y).astype(int)[None, :, None],
                np.linspace(0, x - 1, new_x).astype(int)[None, None, :]
            ]

        elif method == "average":
            # Block-averaging downsample
            # Compute integer block sizes
            bz = int(1 / sz)
            by = int(1 / sy)
            bx = int(1 / sx)

            # Trim to a divisible shape
            tz = (z // bz) * bz
            ty = (y // by) * by
            tx = (x // bx) * bx

            trimmed = self.data[:tz, :ty, :tx]

            # Reshape and average
            ds = trimmed.reshape(
                tz // bz, bz,
                ty // by, by,
                tx // bx, bx
            ).mean(axis=(1, 3, 5))
        else:
            raise ValueError("method must be 'nearest' or 'average'")

        # Then update metadata
        new_meta = dict(self.metadata)

        # Update voxel sizes if present
        if "voxel_size" in new_meta:
            vz, vy, vx = new_meta["voxel_size"]
            new_meta["voxel_size"] = (vz / sz, vy / sy, vx / sx)

        # Metadata for Nrrd and NifTY are a bit differet. See if you have dict names and if so write them
        # NRRD: "space directions"
        if "space directions" in new_meta:
            sd = np.array(new_meta["space directions"])
            sd[:, :] = sd / np.array([sz, sy, sx])[:, None]
            new_meta["space directions"] = sd.tolist()

        # NIfTI: 'pixdim' (index 1,2,3 are x,y,z)
        if "pixdim" in new_meta:
            pix = np.array(new_meta["pixdim"])
            pix[1:4] = pix[1:4] / np.array([sx, sy, sz])
            new_meta["pixdim"] = pix

        return ImagingStack(data=ds, metadata=new_meta)

    # Write method
    # This will call on internal functions
    def to_file(self, filename: str, fmt: Optional[str] = None) -> None:
        """
        Save the ImagingStack to disk as NRRD or NIfTI.

        Parameters
        ----------
        filename : str
            Output file path.
        fmt : str, optional
            "nrrd" or "nifti". If None, inferred from filename.
        """
        lower = filename.lower()

        if fmt is None:
            if lower.endswith(".nrrd"):
                fmt = "nrrd"
            elif lower.endswith(".nii") or lower.endswith(".nii.gz"):
                fmt = "nifti"
            else:
                raise ValueError("Cannot infer format from filename. "
                                 "Use .nrrd, .nii, or .nii.gz or pass fmt='nrrd'/'nifti'.")

        fmt = fmt.lower()
        if fmt == "nrrd":
            self._write_nrrd(filename)
        elif fmt == "nifti":
            self._write_nifti(filename)
        else:
            raise ValueError("fmt must be 'nrrd' or 'nifti'.")

    # Internal function to write NIfTI
    def _write_nifti(self, filename: str) -> None:
        """
        Write current data/metadata as a NIfTI file.
        Tries to convert existing metadata (voxel size, pixdim, space directions)
        into a sensible NIfTI header.
        """
        meta = self.metadata or {}

        # Voxel Sizes
        voxel_size = None

        if "voxel_size" in meta:
            voxel_size = tuple(float(v) for v in meta["voxel_size"])
        elif "pixdim" in meta:
            pixdim = np.array(meta["pixdim"], dtype=float)
            voxel_size = tuple(pixdim[1:4])  # x,y,z spacing
        elif "space directions" in meta:
            sd = np.array(meta["space directions"], dtype=float)
            # length of each direction vector
            voxel_size = tuple(float(np.linalg.norm(v)) for v in sd[:3])
        else:
            voxel_size = (1.0, 1.0, 1.0)

        voxel_size = tuple(voxel_size)

        # NIfTI header & affine
        hdr = nib.Nifti1Header()
        hdr.set_data_dtype(self.data.dtype)

        pixdim = hdr["pixdim"]
        pixdim[1:4] = voxel_size  # NIfTI: pixdim[1..3] = voxel size
        hdr["pixdim"] = pixdim

        affine = np.eye(4, dtype=float)
        affine[0, 0] = voxel_size[0]
        affine[1, 1] = voxel_size[1]
        affine[2, 2] = voxel_size[2]

        img = nib.Nifti1Image(self.data, affine=affine, header=hdr)
        nib.save(img, filename)

    # Internal function to write NRRD
    def _write_nrrd(self, filename: str) -> None:
        """
        Write current data/metadata as a NRRD file.
        Tries to convert existing metadata (voxel size, pixdim) 
        into 'space directions' if needed.
        """
        meta = self.metadata or {}
        header = dict(meta)  # start from existing metadata

        # Ensure 'space directions' is present
        if "space directions" not in header:
            voxel_size = None

            if "voxel_size" in meta:
                voxel_size = tuple(float(v) for v in meta["voxel_size"])
            elif "pixdim" in meta:
                pixdim = np.array(meta["pixdim"], dtype=float)
                voxel_size = tuple(pixdim[1:4])
            else:
                voxel_size = (1.0, 1.0, 1.0)

            vz, vy, vx = voxel_size
            # NRRD uses 3 direction vectors; we'll assume axis-aligned
            header["space directions"] = [
                [vz, 0.0, 0.0],
                [0.0, vy, 0.0],
                [0.0, 0.0, vx],
            ]

        # You can enforce some defaults if you want:
        header.setdefault("encoding", "gzip")
        header.setdefault("endian", "little")

        nrrd.write(filename, self.data, header)
