"""Images in, tensors out. Data URIs and file:// only -- no network fetch."""

import base64
import io
import os
from urllib.parse import unquote

import numpy as np
import openvino as ov
from PIL import Image


def load_image(url_or_data, max_dim):
    """Load an image from a base64 data URI or file:// URI."""
    if url_or_data.startswith("data:"):
        header, b64data = url_or_data.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url_or_data.startswith("file:///"):
        path = unquote(url_or_data[8:])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return Image.open(path).convert("RGB")
    else:
        raise ValueError(
            f"Unsupported image URL scheme. Use data:image/...;base64,... "
            f"or file:///path. Got: {url_or_data[:80]}"
        )


def pil_to_tensor(img, max_dim):
    """Convert PIL Image to OpenVINO Tensor (NHWC uint8)."""
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
        )
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)
