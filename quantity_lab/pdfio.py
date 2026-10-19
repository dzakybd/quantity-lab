"""PDF path: wrap a plan image as an image-only PDF (one PDF point = one image pixel) and render it back."""

from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image


def image_to_pdf(image: Path, out: Path) -> Path:
    """Write `image` as the only content of a PDF page sized width x height points. Not a vector PDF."""
    im = Image.open(image)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    w, h = im.size
    tmp = out.with_suffix(".tmp.png")
    im.save(tmp)
    with pymupdf.open() as doc:
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, filename=str(tmp))
        doc.save(out)
    tmp.unlink()
    return out


def render_rgb(pdf: Path, page: int = 1, dpi: int = 72) -> np.ndarray:
    """Page as an RGB uint8 array (H x W x 3), for the trained model, which reads colour."""
    with pymupdf.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def render_gray(pdf: Path, page: int = 1, dpi: int = 72) -> np.ndarray:
    """Page as a grayscale uint8 array. At 72 DPI one PDF point is one pixel, i.e. the original image resolution."""
    with pymupdf.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
