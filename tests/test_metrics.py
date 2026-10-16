import math

from shapely.geometry import box

from quantity_lab.metrics import compare

SIZE = (400, 200)
GT = [box(10, 10, 190, 190), box(210, 10, 390, 90), box(210, 110, 390, 190)]


def test_identical_rooms_score_perfect():
    m = compare(GT, GT, SIZE)
    assert m["count_f1"] == 1.0 and m["matched"] == 3
    assert m["area_abs_err_mean_px2"] == 0.0 and m["perimeter_abs_err_mean_px"] == 0.0
    assert m["total_area_err_pct"] == 0.0 and m["mask_iou"] == 1.0
    assert m["merged"] == 0 and m["split"] == 0


def test_merged_rooms_detected():
    pred = [GT[0], box(210, 10, 390, 190)]  # the two right rooms as one
    m = compare(pred, GT, SIZE)
    assert m["merged"] == 1
    assert m["matched"] == 1 and math.isclose(m["count_recall"], 1 / 3, abs_tol=1e-3)


def test_split_room_detected():
    pred = [box(10, 10, 90, 190), box(90, 10, 190, 190), GT[1], GT[2]]  # 80/180 and 100/180 of the room
    m = compare(pred, GT, SIZE)
    assert m["split"] == 1
    assert m["matched"] == 3  # the larger half still clears IoU 0.5 (100/180)


def test_shrunk_room_area_error():
    pred = [box(12, 12, 188, 188), GT[1], GT[2]]
    m = compare(pred, GT, SIZE)
    assert m["matched"] == 3
    big = 180 * 180 - 176 * 176
    assert math.isclose(m["area_abs_err_max_px2"], big, rel_tol=0.01)
    assert m["area_signed_err_pct_mean"] < 0


def test_empty_prediction():
    m = compare([], GT, SIZE)
    assert m["matched"] == 0 and m["count_f1"] == 0.0 and m["total_area_err_pct"] == -100.0
