import numpy as np
import pymupdf
from PIL import Image

from quantity_lab.pdfio import image_to_pdf, render_gray


def _img(tmp_path, mode):
    a = np.full((60, 80), 255, np.uint8)
    a[10:50, 20:24] = 0
    im = Image.fromarray(a).convert(mode)
    p = tmp_path / f"p_{mode}.png"
    im.save(p)
    return p


def test_rgb_and_gray_roundtrip(tmp_path):
    for mode in ("RGB", "L", "RGBA"):
        pdf = image_to_pdf(_img(tmp_path, mode), tmp_path / f"{mode}.pdf")
        with pymupdf.open(pdf) as doc:
            assert doc[0].rect == pymupdf.Rect(0, 0, 80, 60)
            assert doc[0].get_drawings() == []
        gray = render_gray(pdf)
        assert gray.shape == (60, 80)
        assert gray[30, 21] < 50 and gray[30, 60] > 200
