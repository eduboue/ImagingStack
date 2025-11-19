import os
import numpy as np
import nibabel as nib
import nrrd
from nd2 import ND2File
from typing import Any, Dict, Iterable, Optional, Tuple, Union 
from numpy.typing import NDArray

class ImagingStack:
    """
    A general container for confocal or multi-channel two-photon imaging stacks.
    Supports up to three imaging channels plus shared metadata and IO helpers.
    """

    def __init__(
        self,
        channel1: NDArray[np.float32],
        channel2: Optional[NDArray[np.float32]] = None,
        channel3: Optional[NDArray[np.float32]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Parameters
        ----------
        channel1 : np.ndarray
            Imaging data for channel 1 (z, y, x).
        metadata : dict, optional
            Arbitrary metadata describing voxel size, acquisition info, etc.
        channel2 : np.ndarray, optional
            Imaging data for channel 2. Must match channel1 shape if provided.
        channel3 : np.ndarray, optional
            Imaging data for channel 3. Must match channel1 shape if provided.
        """
        self.channel1: NDArray[np.float32] = channel1
        self.channel2: Optional[NDArray[np.float32]] = channel2
        self.channel3: Optional[NDArray[np.float32]] = channel3
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self._validate_channels()

    @property
    def data(self) -> NDArray[np.float32]:
        """Backward compatible alias for channel 1."""
        return self.channel1

    @data.setter
    def data(self, value: NDArray[np.float32]) -> None:
        self.channel1 = value
        self._validate_channels()

    # General class method to load the class.
    # classmethod will read files name, and identify the FileType
    # Then, it calls on one of three helper function, depending on
    # FileType.
    @classmethod
    def from_file(
        cls,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImagingStack":
        """
        Detect whether the file is NRRD, NIfTI, or ND2 and load an ImagingStack.

        Parameters
        ----------
        filename : str
            Path to the imaging file.
        metadata : dict, optional
            Extra metadata to merge into the loaded header information.

        Returns
        -------
        ImagingStack
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File does not exist: {filename}")

        # Determine file extension
        fname = filename.lower()

        if fname.endswith(".nrrd"):
            (ch1, ch2, ch3), meta = cls._load_nrrd(filename)
        elif fname.endswith(".nii") or fname.endswith(".nii.gz"):
            (ch1, ch2, ch3), meta = cls._load_nifti(filename)
        elif fname.endswith(".nd2"):
            (ch1, ch2, ch3), meta = cls._load_nd2(filename)
        else:
            raise ValueError(
                "Unsupported file type. Must be .nrrd, .nii, .nii.gz, or .nd2"
            )

        merged_meta = dict(meta)
        if metadata:
            merged_meta.update(metadata)

        return cls(channel1=ch1, channel2=ch2, channel3=ch3, metadata=merged_meta)

    # General dunder to get data info
    def __repr__(self) -> str:
        """Return a concise string summary describing channel shapes and metadata keys."""
        channel_desc: Iterable[str] = (
            f"C{idx}:{ch.shape}" for idx, ch in self._enumerate_channels() if ch is not None
        )
        channel_str = ", ".join(channel_desc) or "no channels"
        return f"ImagingStack({channel_str}, metadata_keys={list(self.metadata.keys())})"
    
    # Downsample function
    def downsample(
        self, 
        scale: Union[float, Tuple[float, float, float]] = 0.25,
        method: str = "nearest"
        ) -> "ImagingStack":
        """
        Downsample the image volume by a scale factor for every channel.
        
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
            A new ImagingStack with downsampled channels and updated metadata.
        """
        if isinstance(scale, (int, float)):
            sz = sy = sx = float(scale)
        else:
            if len(scale) != 3:
                raise ValueError("scale must be float or 3-tuple (z, y, x)")
            sz, sy, sx = map(float, scale)

        if method not in {"nearest", "average"}:
            raise ValueError("method must be 'nearest' or 'average'")

        new_channels = [
            self._downsample_channel(channel, sz, sy, sx, method)
            if channel is not None else None
            for channel in self._iter_channels()
        ]

        new_meta = self._scaled_metadata(sz, sy, sx)

        return ImagingStack(
            channel1=new_channels[0],
            channel2=new_channels[1],
            channel3=new_channels[2],
            metadata=new_meta,
        )

    # Write method
    # This will call on internal functions
    def to_file(
        self,
        filename: str,
        fmt: Optional[str] = None,
        channel: int = 1,
        compress_nifti: bool = True,
    ) -> None:
        """
        Save a single channel of the ImagingStack to disk as NRRD or NIfTI.

        Parameters
        ----------
        filename : str
            Output file path.
        fmt : str, optional
            "nrrd" or "nifti". If None, inferred from filename.
        channel : int, optional
            Channel to write (1-3). The output filename is suffixed with `_0{channel}`.
        compress_nifti : bool, optional
            When writing NIfTI, set True to default to `.nii.gz` output, False for `.nii`.
        """
        data = self._get_channel(channel)
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
        target_path = self._apply_channel_suffix(filename, channel)

        if fmt == "nrrd":
            self._write_nrrd(target_path, data)
        elif fmt == "nifti":
            target_path = self._normalize_nifti_extension(target_path, compress_nifti)
            self._write_nifti(target_path, data)
        else:
            raise ValueError("fmt must be 'nrrd' or 'nifti'.")

    @staticmethod
    def _load_nrrd(filename: str) -> Tuple[Tuple[NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]], Dict[str, Any]]:
        data, header = nrrd.read(filename)
        metadata = dict(header)
        channels = ImagingStack._split_channels_default(np.asarray(data, dtype=np.float32))
        return channels, metadata

    @staticmethod
    def _load_nifti(filename: str) -> Tuple[Tuple[NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]], Dict[str, Any]]:
        img = nib.load(filename)
        data = img.get_fdata(dtype=np.float32)
        metadata = dict(img.header)
        channels = ImagingStack._split_channels_default(data)
        return channels, metadata

    @staticmethod
    def _load_nd2(filename: str) -> Tuple[Tuple[NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]], Dict[str, Any]]:
        with ND2File(filename) as nd_file:
            data = nd_file.asarray()
            axes = ImagingStack._determine_nd2_axes(nd_file)
            nd_meta = getattr(nd_file, "metadata", None)

        channels = ImagingStack._split_channels_from_nd2(np.asarray(data, dtype=np.float32), axes)
        metadata: Dict[str, Any] = {
            "source_file": filename,
        }
        if axes:
            metadata["nd2_axes"] = axes
        if nd_meta is not None:
            metadata["nd2_metadata"] = str(nd_meta)
        return channels, metadata

    @staticmethod
    def _determine_nd2_axes(nd_file: ND2File) -> str:
        """
        Best-effort helper to recover the axis specification from an ND2 file.
        Older nd2-reader versions sometimes omit the `axes` attribute, in which
        case we fall back to constructing it from the ordered sizes mapping.
        """
        axes = getattr(nd_file, "axes", None) or ""
        if axes:
            return axes

        sizes = getattr(nd_file, "sizes", None)
        if isinstance(sizes, dict) and sizes:
            return "".join(sizes.keys())

        return ""

    # Internal function to write NIfTI
    def _write_nifti(self, filename: str, data: NDArray[np.float32]) -> None:
        """
        Write a single channel/metadata as a NIfTI file.
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
        hdr.set_data_dtype(data.dtype)

        pixdim = hdr["pixdim"]
        pixdim[1:4] = voxel_size  # NIfTI: pixdim[1..3] = voxel size
        hdr["pixdim"] = pixdim

        affine = np.eye(4, dtype=float)
        affine[0, 0] = voxel_size[0]
        affine[1, 1] = voxel_size[1]
        affine[2, 2] = voxel_size[2]

        img = nib.Nifti1Image(data, affine=affine, header=hdr)
        nib.save(img, filename)

    # Internal function to write NRRD
    def _write_nrrd(self, filename: str, data: NDArray[np.float32]) -> None:
        """
        Write a single channel/metadata as a NRRD file.
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

        nrrd.write(filename, data, header)
        
    def _iter_channels(self) -> Tuple[Optional[NDArray[np.float32]], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]]:
        return (self.channel1, self.channel2, self.channel3)

    def _enumerate_channels(self):
        return enumerate(self._iter_channels(), start=1)

    def _get_channel(self, channel: int) -> NDArray[np.float32]:
        if channel not in {1, 2, 3}:
            raise ValueError("channel must be 1, 2, or 3")
        data = self._iter_channels()[channel - 1]
        if data is None:
            raise ValueError(f"Channel {channel} is empty.")
        return data

    def _validate_channels(self) -> None:
        base_shape = self.channel1.shape
        for idx, channel in enumerate((self.channel2, self.channel3), start=2):
            if channel is not None and channel.shape != base_shape:
                raise ValueError(
                    f"Channel {idx} shape {channel.shape} does not match channel 1 {base_shape}."
                )

    @staticmethod
    def _split_channels_default(
        data: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]]:
        if data.ndim == 4 and data.shape[-1] <= 3:
            parts = [np.asarray(data[..., idx]) for idx in range(data.shape[-1])]
        else:
            parts = [np.asarray(data)]

        while len(parts) < 3:
            parts.append(None)

        return parts[0], parts[1], parts[2]

    @staticmethod
    def _split_channels_from_nd2(
        data: NDArray[np.float32],
        axes: str,
    ) -> Tuple[NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]]:
        axes = axes.upper() if axes else ""
        if axes:
            selectors = [slice(None)] * data.ndim
            for idx, axis_name in enumerate(axes):
                if axis_name not in {"C", "Z", "Y", "X"}:
                    selectors[idx] = 0
            trimmed = data[tuple(selectors)]
            trimmed_axes = "".join(axis for axis in axes if axis in {"C", "Z", "Y", "X"})

            channel_axis = trimmed_axes.find("C")
            if channel_axis == -1:
                channel_slices = [trimmed]
                axis_without_channel = trimmed_axes
            else:
                num_channels = min(3, trimmed.shape[channel_axis])
                channel_slices = [
                    np.take(trimmed, idx, axis=channel_axis) for idx in range(num_channels)
                ]
                axis_without_channel = trimmed_axes.replace("C", "")

            zyx_parts = [
                ImagingStack._ensure_zyx_orientation(slice_arr, axis_without_channel)
                for slice_arr in channel_slices
            ]

            while len(zyx_parts) < 3:
                zyx_parts.append(None)

            return zyx_parts[0], zyx_parts[1], zyx_parts[2]

        # Fallback: rely on rank only (for ND2 builds that don't populate axes metadata).
        if data.ndim == 3:
            ch1 = np.asarray(data)
            return ch1, None, None

        if data.ndim == 4:
            # Assume axis order (Z, C, Y, X) and split along the second dimension.
            num_channels = min(3, data.shape[1])
            zyx_parts = [np.asarray(data[:, idx, ...]) for idx in range(num_channels)]
            while len(zyx_parts) < 3:
                zyx_parts.append(None)
            return zyx_parts[0], zyx_parts[1], zyx_parts[2]

        raise ValueError(
            f"Unsupported ND2 array shape {data.shape}; unable to infer channels without axis metadata."
        )

    @staticmethod
    def _ensure_zyx_orientation(data: NDArray[np.float32], axes: str) -> NDArray[np.float32]:
        axes = axes.upper()
        required = ("Z", "Y", "X")
        if any(axis not in axes for axis in required):
            raise ValueError(f"ND2 data missing required spatial axes: {axes}")

        axis_positions = {axis: idx for idx, axis in enumerate(axes)}
        order = [axis_positions["Z"], axis_positions["Y"], axis_positions["X"]]
        return np.asarray(np.transpose(data, order))

    @staticmethod
    def _downsample_channel(
        data: NDArray[np.float32],
        sz: float,
        sy: float,
        sx: float,
        method: str,
    ) -> NDArray[np.float32]:
        z, y, x = data.shape[:3]

        new_z = max(1, int(z * sz))
        new_y = max(1, int(y * sy))
        new_x = max(1, int(x * sx))

        if method == "nearest":
            return data[
                np.linspace(0, z - 1, new_z).astype(int)[:, None, None],
                np.linspace(0, y - 1, new_y).astype(int)[None, :, None],
                np.linspace(0, x - 1, new_x).astype(int)[None, None, :],
            ]
        elif method == "average":
            bz = int(1 / sz)
            by = int(1 / sy)
            bx = int(1 / sx)

            tz = (z // bz) * bz
            ty = (y // by) * by
            tx = (x // bx) * bx

            trimmed = data[:tz, :ty, :tx]

            return trimmed.reshape(
                tz // bz,
                bz,
                ty // by,
                by,
                tx // bx,
                bx,
            ).mean(axis=(1, 3, 5))

        raise ValueError(f"Unsupported downsample method '{method}' in _downsample_channel")

    def _scaled_metadata(self, sz: float, sy: float, sx: float) -> Dict[str, Any]:
        new_meta = dict(self.metadata)

        if "voxel_size" in new_meta:
            vz, vy, vx = new_meta["voxel_size"]
            new_meta["voxel_size"] = (vz / sz, vy / sy, vx / sx)

        if "space directions" in new_meta:
            sd = np.array(new_meta["space directions"])
            sd[:, :] = sd / np.array([sz, sy, sx])[:, None]
            new_meta["space directions"] = sd.tolist()

        if "pixdim" in new_meta:
            pix = np.array(new_meta["pixdim"])
            pix[1:4] = pix[1:4] / np.array([sx, sy, sz])
            new_meta["pixdim"] = pix

        return new_meta

    @staticmethod
    def _apply_channel_suffix(filename: str, channel: int) -> str:
        if channel not in {1, 2, 3}:
            raise ValueError("channel must be 1, 2, or 3")

        base, ext = os.path.splitext(filename)
        if ext.lower() == ".gz" and base.lower().endswith(".nii"):
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext

        return f"{base}_{channel:02d}{ext}"

    @staticmethod
    def _normalize_nifti_extension(filename: str, compress: bool) -> str:
        lower = filename.lower()

        if compress:
            if lower.endswith(".nii.gz"):
                return filename
            if lower.endswith(".nii"):
                filename = filename[: -len(".nii")]
            return f"{filename}.nii.gz"

        # not compress: ensure plain .nii
        if lower.endswith(".nii.gz"):
            filename = filename[: -len(".gz")]
        if not filename.lower().endswith(".nii"):
            filename += ".nii"
        return filename
