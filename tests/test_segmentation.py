import numpy as np

from quantity_lab.segmentation import DOOR, KITCHEN, LIVING, OTHER, OUTDOOR, WALL, SegParams, rooms_from_maps


def _maps():
    """Two rooms of different classes behind one wall with a doorway, an outdoor terrace and a door icon."""
    rooms = np.full((100, 200), LIVING, np.int64)
    rooms[:, 100:] = KITCHEN
    rooms[:, 98:102] = WALL
    rooms[40:60, 98:102] = LIVING  # the doorway: the model sees floor, not wall
    rooms[:, :5] = OUTDOOR
    icons = np.zeros_like(rooms)
    icons[40:60, 98:102] = DOOR
    return rooms, icons


def test_walls_alone_join_rooms_through_a_doorway():
    lab = rooms_from_maps(*_maps(), SegParams(close_doors=False, split_by_class=False, min_room_share=0.005))
    assert len(set(np.unique(lab)) - {0}) == 1


def test_door_icons_close_the_doorway():
    lab = rooms_from_maps(*_maps(), SegParams(close_doors=True, split_by_class=False, min_room_share=0.005))
    assert len(set(np.unique(lab)) - {0}) == 2
    assert lab[50, 50] != lab[50, 150]


def test_room_class_splits_an_open_doorway():
    lab = rooms_from_maps(*_maps(), SegParams(close_doors=False, split_by_class=True, min_room_share=0.005))
    assert lab[50, 50] != lab[50, 150] and lab[50, 50] != 0 and lab[50, 150] != 0


def test_outdoor_wall_and_small_pieces_are_not_rooms():
    rooms, icons = _maps()
    rooms[10:13, 150:153] = OTHER  # a 9 px speck of another class inside the kitchen
    lab = rooms_from_maps(rooms, icons, SegParams(close_doors=True, split_by_class=True, min_room_share=0.005))
    assert lab[50, 2] == 0  # outdoor
    assert lab[0, 99] == 0  # wall
    assert lab[11, 151] == 0  # too small to be a room


def test_params_roundtrip():
    p = SegParams(close_doors=False, split_by_class=True, min_room_share=0.0050)
    assert SegParams.from_dict(p.to_dict()) == p
