import numpy as np

from quantity_lab.baseline import BaselineParams, segment_rooms


def _plan():
    g = np.full((220, 320), 255, np.uint8)
    t = 8
    g[10 : 10 + t, 10:310] = 0  # outer walls
    g[202:210, 10:310] = 0
    g[10:210, 10:18] = 0
    g[10:210, 302:310] = 0
    g[10:210, 156:164] = 0  # divider with a door gap
    g[90:130, 156:164] = 255
    g[50:51, 40:120] = 0  # thin furniture line inside the left room
    return g


def test_two_rooms_found_through_door_gap():
    lab = segment_rooms(_plan(), BaselineParams(wall_px=5, door_px=50, min_room_px2=500, work_long_side=0))
    ids = set(np.unique(lab)) - {0}
    assert len(ids) == 2
    assert lab[60, 80] != 0 and lab[60, 80] != lab[60, 230]


def test_outside_is_not_a_room():
    lab = segment_rooms(_plan(), BaselineParams(wall_px=5, door_px=50, min_room_px2=500, work_long_side=0))
    assert lab[2, 2] == 0 and lab[215, 315] == 0


def test_params_roundtrip():
    p = BaselineParams(threshold=100)
    assert BaselineParams.from_dict(p.to_dict()) == p


def test_narrow_corridor_not_filled_by_door_closing():
    g = np.full((260, 400), 255, np.uint8)
    for sl in [
        (slice(10, 18), slice(10, 390)),
        (slice(242, 250), slice(10, 390)),
        (slice(10, 250), slice(10, 18)),
        (slice(10, 250), slice(382, 390)),
        (slice(10, 250), slice(150, 158)),
        (slice(10, 250), slice(210, 218)),
    ]:
        g[sl] = 0  # two rooms with a 52 px corridor between them
    lab = segment_rooms(g, BaselineParams(wall_px=5, door_px=90, min_room_px2=500, work_long_side=0))
    assert lab[130, 184] != 0  # the corridor is a room, not wall
    assert len(set(np.unique(lab)) - {0}) == 3


def test_working_resolution_keeps_output_size():
    g = np.kron(_plan(), np.ones((3, 3), np.uint8))  # same plan at 3x the resolution
    lab = segment_rooms(g, BaselineParams(wall_px=5, door_px=50, min_room_px2=500, work_long_side=320))
    assert lab.shape == g.shape
    assert len(set(np.unique(lab)) - {0}) == 2
