import unittest

from planner import (
    build_drag_route,
    build_field_route,
    estimate_camera_translation,
    select_center_match,
    select_field_center,
    select_nearest_match,
    translate_points,
)


class PlannerTests(unittest.TestCase):
    def test_selects_crop_nearest_group_center(self):
        matches = [(0, 0, 10, 10), (40, 40, 10, 10), (100, 100, 10, 10)]

        self.assertEqual(select_center_match(matches), matches[1])

    def test_selects_match_nearest_given_point(self):
        matches = [(100, 100, 10, 10), (30, 25, 10, 10), (400, 300, 10, 10)]

        self.assertEqual(select_nearest_match(matches, (20, 20)), matches[1])

    def test_nearest_match_rejects_a_match_far_from_the_click(self):
        # A tool icon genuinely tied to a click should be nearby. A match
        # far away (e.g. an unrelated HUD icon in a screen corner) must be
        # rejected rather than dragged to, however high its confidence was.
        matches = [(1635, 168, 30, 30)]

        self.assertIsNone(select_nearest_match(matches, (780, 760), max_distance=500))
        self.assertEqual(select_nearest_match(matches, (780, 760)), matches[0])

    def test_field_center_is_none_without_matches(self):
        self.assertIsNone(select_field_center([]))

    def test_field_center_uses_bounding_box_midpoint_not_a_raw_match(self):
        # Only the two opposite corners of a field were matched (the tiles
        # in between fell under the threshold). Picking either raw match as
        # the click point lands right on the field's edge; the bounding-box
        # midpoint stays inside the field regardless of which tiles matched.
        matches = [(940, 580, 20, 20), (790, 710, 20, 20)]

        self.assertEqual(select_field_center(matches), (865, 645))
        self.assertNotIn(select_field_center(matches), [(m[0], m[1]) for m in matches])

    def test_uses_shared_anchor_translation(self):
        result = estimate_camera_translation(
            {"boat": (100, 200)},
            {"boat": (112, 193)},
            [],
            [],
            (1920, 1080),
        )

        self.assertEqual(
            (result.dx, result.dy, result.method, result.confidence),
            (12, -7, "shared_anchor", "boat"),
        )

    def test_uses_median_crop_translation_without_anchors(self):
        result = estimate_camera_translation(
            {},
            {},
            [(100, 100), (200, 100), (300, 100)],
            [(107, 96), (207, 96), (307, 96)],
            (1920, 1080),
        )

        self.assertEqual((result.dx, result.dy, result.method), (7, -4, "crop_positions"))
        self.assertEqual(result.confidence, "3/3_pairs")

    def test_falls_back_to_stable_scene_without_movement_evidence(self):
        result = estimate_camera_translation({}, {}, [(10, 10)], [], (1920, 1080))

        self.assertEqual(
            (result.dx, result.dy, result.method),
            (0, 0, "stable_fallback"),
        )

    def test_rejects_translation_larger_than_screen_limit(self):
        result = estimate_camera_translation(
            {"boat": (100, 100)},
            {"boat": (700, 100)},
            [],
            [],
            (1920, 1080),
        )

        self.assertEqual(
            (result.dx, result.dy, result.method, result.confidence),
            (0, 0, "stable_fallback", "rejected_shared_anchor"),
        )

    def test_translates_points_with_camera_shift(self):
        translation = estimate_camera_translation(
            {"market": (50, 50)},
            {"market": (45, 58)},
            [],
            [],
            (1920, 1080),
        )

        self.assertEqual(translate_points([(100, 200), (300, 400)], translation), [(95, 208), (295, 408)])

    def test_one_crop_route_reaches_exact_center(self):
        self.assertEqual(
            build_drag_route([(50, 0)], (0, 0), max_segment=25),
            [(25, 0), (50, 0)],
        )

    def test_route_visits_nearest_crop_first(self):
        route = build_drag_route(
            [(100, 0), (20, 0), (60, 0)],
            (0, 0),
            max_segment=100,
        )

        self.assertEqual(route, [(20, 0), (60, 0), (100, 0)])

    def test_route_removes_duplicate_crop_centers(self):
        route = build_drag_route([(20, 0), (20, 0)], (0, 0), max_segment=100)

        self.assertEqual(route, [(20, 0)])

    def test_field_route_is_empty_without_matches(self):
        self.assertEqual(build_field_route([], (0, 0)), [])

    def test_field_route_sweeps_rows_of_real_points_without_backtracking(self):
        # A nearest-neighbor route can zig-zag back across a field, producing
        # short back-and-forth segments that look like jitter to touch input.
        # Real tiles must be visited row by row in one consistent direction.
        matches = [
            (100, 100, 20, 20), (150, 100, 20, 20), (200, 100, 20, 20),
            (100, 140, 20, 20), (150, 140, 20, 20), (200, 140, 20, 20),
        ]

        route = build_field_route(matches, start=(100, 90), max_segment=1000)

        self.assertEqual(
            route,
            [
                (100, 100), (150, 100), (200, 100),
                (200, 140), (150, 140), (100, 140),
            ],
        )

    def test_field_route_starts_from_the_row_nearest_the_start_point(self):
        matches = [
            (100, 100, 20, 20), (200, 100, 20, 20),
            (100, 140, 20, 20), (200, 140, 20, 20),
        ]

        route = build_field_route(matches, start=(100, 150), max_segment=1000)

        self.assertEqual(
            route,
            [(100, 140), (200, 140), (200, 100), (100, 100)],
        )

    def test_field_route_row_start_overrides_which_row_sweeps_first(self):
        matches = [
            (100, 100, 20, 20), (150, 100, 20, 20), (200, 100, 20, 20),
            (100, 140, 20, 20), (150, 140, 20, 20), (200, 140, 20, 20),
        ]

        # start (where the drag physically begins, e.g. the tool icon) is
        # near the bottom row, but row_start (the field tile the user
        # actually selected) is near the top row -- the sweep must begin
        # at the selected tile's row, not wherever the tool icon sits.
        route = build_field_route(
            matches,
            start=(100, 150),
            row_start=(100, 90),
            max_segment=1000,
        )

        self.assertEqual(
            route,
            [
                (100, 100), (150, 100), (200, 100),
                (200, 140), (150, 140), (100, 140),
            ],
        )


if __name__ == "__main__":
    unittest.main()
