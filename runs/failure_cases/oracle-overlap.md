# One reference room disappears in the label-mask round trip

Plan high_quality/5632 has nine indoor room annotations. Its seventh room (Undefined, area 2402.72 px²) has zero pixels in the instance label mask because a later overlapping annotation covers it. The rasterizer explicitly uses the last room on overlaps. The round trip therefore has eight rooms, count F1 0.9412, while mask IoU remains 1.0. This is an annotation and representation ambiguity, not a failure of Shapely's area calculation or a subpixel room.

Keep this plan in the benchmark. `category_map.json` records overlap area and fully occluded room IDs; `tests/test_geometry.py::test_overlapping_annotation_label_mask_records_last_room` pins the overlap rule. Per-room reference quantities measure each annotated polygon, while total area is the sum of those polygon areas, not their union. They must not be presented as non-overlapping physical floor area.
