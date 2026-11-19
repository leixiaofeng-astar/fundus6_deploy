"""Decode one fundus image; reject ambiguous DICOM frame/colour layouts."""
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def read_image(path):
    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as im:
                if im.width * im.height > MAX_PIXELS or getattr(im, 'n_frames', 1) != 1:
                    raise ValueError('Only single-frame images up to 25 megapixels are supported')
                im.load()
                return ImageOps.exif_transpose(im).convert('RGB')
    except UnidentifiedImageError:
        pass
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as e:
        raise ValueError('Image exceeds pixel limit') from e

    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut, convert_color_space
    try:
        ds = pydicom.dcmread(path)  # no force: arbitrary uploads are not DICOM
        rows, cols = int(ds.Rows), int(ds.Columns)
        if rows * cols > MAX_PIXELS or int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Only single-frame DICOM up to 25 megapixels is supported')
        if int(ds.BitsAllocated) not in (8, 16) or int(ds.SamplesPerPixel) not in (1, 3):
            raise ValueError('Unsupported DICOM bit depth/sample count')
        arr = ds.pixel_array  # pydicom converts YBR_FULL to RGB
        if arr.ndim == 3 and arr.shape[-1] == 3:
            colour = str(ds.PhotometricInterpretation)
            if colour not in ('RGB', 'YBR_FULL', 'YBR_FULL_422'):
                raise ValueError('Unsupported DICOM colour space')
            if int(pydicom.__version__.split('.')[0]) < 3 and colour.startswith('YBR'):
                arr = convert_color_space(arr, colour, 'RGB')
            if arr.dtype != np.uint8:
                raise ValueError('Only 8-bit colour DICOM is supported')
            return Image.fromarray(arr).convert('RGB')
        if arr.ndim != 2 or ds.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2'):
            raise ValueError('Unsupported DICOM pixel layout')
        arr = np.asarray(apply_voi_lut(apply_modality_lut(arr, ds), ds), dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError('Invalid DICOM pixel values')
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1 - arr
        return Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
    except Exception as e:
        raise ValueError('Invalid/unsupported image or DICOM; compressed DICOM may need pylibjpeg plugins') from e
