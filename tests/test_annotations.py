from pathlib import Path

import pytest

from quantity_lab.annotations import load_plan

PLAN = Path(__file__).parent / "fixtures" / "plan.svg"


def test_rooms_walls_and_counts():
    plan = load_plan(PLAN)
    assert plan.size == (300, 200)
    assert [r.label for r in plan.rooms] == ["Bedroom", "Kitchen", "Outdoor", "Undefined"]
    assert plan.rooms[0].polygon.area == 120 * 160
    assert len(plan.walls) == 2
    assert plan.door_count == 1 and plan.window_count == 1


def test_outdoor_flagged_and_invalid_room_repaired():
    plan = load_plan(PLAN)
    assert [r.outdoor for r in plan.rooms] == [False, False, True, False]
    bowtie = plan.rooms[3]
    assert bowtie.repaired and bowtie.polygon.is_valid


def test_indoor_rooms_helper():
    assert [r.label for r in load_plan(PLAN).indoor()] == ["Bedroom", "Kitchen", "Undefined"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_plan(Path("nope/model.svg"))


def test_empty_plan(tmp_path):
    p = tmp_path / "e.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')
    plan = load_plan(p)
    assert plan.rooms == [] and plan.door_count == 0
