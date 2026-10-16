import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box

from quantity_lab.geometry import measure, polygonize, rasterize, repair


def test_square_area_and_perimeter():
    assert measure(box(0, 0, 10, 10)) == (100.0, 40.0)


def test_scaling_by_two():
    a, p = measure(affinity.scale(box(0, 0, 10, 10), 2, 2, origin=(0, 0)))
    assert (a, p) == (400.0, 80.0)


@settings(max_examples=50)
@given(st.floats(-500, 500), st.floats(-500, 500), st.floats(0, 360))
def test_rigid_motion_preserves_area_and_perimeter(dx, dy, deg):
    poly = Polygon([(0, 0), (40, 0), (40, 25), (15, 25), (15, 60), (0, 60)])
    moved = affinity.translate(affinity.rotate(poly, deg, origin=(0, 0)), dx, dy)
    a0, p0 = measure(poly)
    a1, p1 = measure(moved)
    assert math.isclose(a0, a1, rel_tol=1e-9) and math.isclose(p0, p1, rel_tol=1e-9)


@settings(max_examples=50)
@given(
    st.lists(
        st.tuples(st.floats(0, 100), st.floats(0, 100), st.floats(1, 50), st.floats(1, 50)), min_size=2, max_size=5
    )
)
def test_union_area_not_above_sum(boxes):
    polys = [box(x, y, x + w, y + h) for x, y, w, h in boxes]
    from shapely.ops import unary_union

    assert unary_union(polys).area <= sum(p.area for p in polys) + 1e-6


def test_repair_self_intersection_keeps_flag():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    fixed, repaired = repair(bowtie)
    assert repaired and fixed.is_valid
    assert isinstance(fixed, (Polygon, MultiPolygon))
    assert math.isclose(fixed.area, 50.0)


def test_repair_valid_polygon_untouched():
    p = box(0, 0, 5, 5)
    fixed, repaired = repair(p)
    assert not repaired and fixed.equals(p)


def test_raster_roundtrip_area_close():
    rooms = [box(10, 10, 110, 60), box(120, 10, 200, 90)]
    mask = rasterize(rooms, (220, 120))
    assert mask.dtype == np.int32 and set(np.unique(mask)) == {0, 1, 2}
    back = polygonize(mask)
    assert len(back) == 2
    for orig, (label, poly) in zip(rooms, sorted(back.items())):
        assert abs(poly.area - orig.area) / orig.area < 0.03


def test_polygonize_keeps_holes():
    ring = box(0, 0, 100, 100).difference(box(40, 40, 60, 60))
    back = polygonize(rasterize([ring], (120, 120)))
    assert abs(back[1].area - ring.area) / ring.area < 0.05


def test_tiny_room_survives():
    back = polygonize(rasterize([box(5, 5, 9, 9)], (20, 20)))
    assert len(back) == 1 and back[1].area > 0
