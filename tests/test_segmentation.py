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


def _two_rooms_one_doorway():
    """Two same-class rooms joined by a 10 px doorway: walls alone see one region."""
    rooms = np.full((120, 260), WALL, np.int64)
    rooms[10:110, 10:125] = LIVING
    rooms[10:110, 135:250] = LIVING
    rooms[55:65, 125:135] = LIVING  # the doorway
    return rooms, np.zeros_like(rooms)


def test_neck_split_separates_rooms_at_a_doorway():
    maps = _two_rooms_one_doorway()
    whole = rooms_from_maps(
        *maps, SegParams(close_doors=False, split_by_class=False, min_room_share=0.01, neck_share=0.0)
    )
    assert len(set(np.unique(whole)) - {0}) == 1
    lab = rooms_from_maps(
        *maps, SegParams(close_doors=False, split_by_class=False, min_room_share=0.01, neck_share=0.08)
    )
    assert len(set(np.unique(lab)) - {0}) == 2
    assert lab[60, 60] != lab[60, 200] and lab[60, 60] != 0 and lab[60, 200] != 0
    assert (lab > 0).sum() == (whole > 0).sum()  # splitting moves no floor in or out


def test_neck_split_leaves_a_plain_room_whole():
    rooms = np.full((120, 160), WALL, np.int64)
    rooms[10:110, 10:150] = KITCHEN
    lab = rooms_from_maps(
        rooms, np.zeros_like(rooms), SegParams(split_by_class=False, min_room_share=0.01, neck_share=0.08)
    )
    assert len(set(np.unique(lab)) - {0}) == 1


def test_neck_split_keeps_each_room_whole_up_to_its_walls():
    maps = _two_rooms_one_doorway()
    lab = rooms_from_maps(
        *maps, SegParams(close_doors=False, split_by_class=False, min_room_share=0.01, neck_share=0.08)
    )
    left, right = lab[60, 60], lab[60, 200]
    assert (lab[10:110, 10:125] == left).all()  # the whole left room, including the pixels next to its walls
    assert (lab[10:110, 135:250] == right).all()
