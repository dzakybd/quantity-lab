import numpy as np
from shapely.geometry import box

from quantity_lab.geometry import rasterize
from quantity_lab.perturb import dilate, erode, merge_adjacent, scale, shift, split_largest

SIZE = (300, 200)
ROOMS = [box(10, 10, 140, 190), box(150, 10, 290, 190)]  # 10 px wall between


def test_erode_and_dilate_change_area_by_perimeter():
    m = rasterize(ROOMS, SIZE)
    a0 = (m == 1).sum()
    assert (erode(m, 2) == 1).sum() == (130 - 4) * (180 - 4)
    assert (dilate(m, 2) == 1).sum() > a0


def test_dilate_does_not_overwrite_other_rooms():
    m = rasterize(ROOMS, SIZE)
    d = dilate(m, 8)
    assert (d == 2).sum() >= (m == 2).sum()


def test_shift_moves_mask():
    m = rasterize(ROOMS, SIZE)
    s = shift(m, 5, 0)
    assert (s == 1).sum() == (m == 1).sum() and s[100, 12] == 0 and s[100, 16] == 1


def test_scale_multiplies_area():
    m = rasterize(ROOMS, SIZE)
    s = scale(m, 1.02)
    assert abs((s == 1).sum() / (m == 1).sum() - 1.02**2) < 0.01


def test_merge_adjacent_joins_two_rooms():
    out = merge_adjacent(rasterize(ROOMS, SIZE), gap=12)
    assert len(set(np.unique(out)) - {0}) == 1
    assert (out > 0).sum() > (rasterize(ROOMS, SIZE) > 0).sum()  # the divider becomes floor


def test_split_largest_makes_two_pieces():
    out = split_largest(rasterize(ROOMS, SIZE), width=3)
    assert len(set(np.unique(out)) - {0}) == 3
