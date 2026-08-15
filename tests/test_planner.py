import unittest

from planner import (
    build_drag_route,
    estimate_camera_translation,
    select_center_match,
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


if __name__ == "__main__":
    unittest.main()
