import unittest
from unittest.mock import patch

import cv2
import numpy as np

from matcher import Matcher, TemplateMatchResult


class MatcherTests(unittest.TestCase):
    def test_match_details_include_matches_and_best_confidence(self):
        target = np.zeros((30, 30), dtype=np.uint8)
        template = np.array(
            [
                [1, 7, 4, 9, 2],
                [6, 3, 8, 2, 5],
                [9, 2, 5, 1, 7],
                [4, 8, 3, 6, 1],
                [2, 5, 9, 4, 8],
            ],
            dtype=np.uint8,
        )
        target[10:15, 12:17] = template

        result = Matcher().match_template_details(
            template,
            target,
            matching_threshold=0.999,
            grouping=False,
        )

        self.assertIsInstance(result, TemplateMatchResult)
        self.assertGreaterEqual(result.best_confidence, 0.999)
        self.assertIn([14, 12, 5, 5], result.matches)

    def test_match_template_keeps_list_return_type(self):
        target = np.zeros((20, 20), dtype=np.uint8)
        template = np.array([[1, 2], [3, 7]], dtype=np.uint8)
        target[6:8, 9:11] = template

        matches = Matcher().match_template(
            template,
            target,
            matching_threshold=0.999,
            grouping=False,
        )

        self.assertIsInstance(matches, list)
        self.assertIn([10, 7, 2, 2], matches)

    def test_grouping_keeps_one_isolated_valid_match(self):
        target = np.zeros((20, 20), dtype=np.uint8)
        template = np.array([[1, 2], [3, 7]], dtype=np.uint8)
        target[6:8, 9:11] = template

        with patch.object(
            __import__("matcher").cv2,
            "groupRectangles",
            return_value=([], []),
            create=True,
        ):
            result = Matcher().match_template_details(
                template,
                target,
                matching_threshold=0.999,
                grouping=True,
            )

        self.assertEqual(result.matches, [[10, 7, 2, 2]])

    def test_match_details_report_raw_count_and_best_location(self):
        target = np.zeros((20, 20), dtype=np.uint8)
        template = np.array([[1, 2], [3, 7]], dtype=np.uint8)
        target[6:8, 9:11] = template

        result = Matcher().match_template_details(
            template,
            target,
            matching_threshold=0.999,
            grouping=True,
        )

        self.assertEqual(result.raw_match_count, 1)
        self.assertEqual(result.best_match, [10, 7, 2, 2])

    def test_multiscale_matching_finds_resized_template_at_screen_coordinates(self):
        template = np.array(
            [
                [3, 40, 7, 90, 12, 63, 5, 31],
                [70, 8, 55, 2, 84, 18, 46, 9],
                [6, 95, 21, 74, 11, 38, 82, 4],
                [44, 13, 67, 25, 91, 3, 52, 78],
                [19, 86, 1, 58, 34, 73, 16, 49],
                [92, 24, 61, 10, 80, 29, 65, 14],
                [35, 76, 17, 88, 6, 54, 27, 69],
                [81, 15, 47, 33, 71, 20, 96, 8],
            ],
            dtype=np.uint8,
        )
        scaled = cv2.resize(
            template,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )
        target = np.zeros((60, 60), dtype=np.uint8)
        target[20:32, 30:42] = scaled

        result = Matcher().match_template_multiscale(
            template,
            target,
            matching_threshold=0.999,
            scales=(1.0, 1.5),
        )

        self.assertEqual(result.template_scale, 1.5)
        self.assertEqual(result.matches, [[36, 26, 12, 12]])
        self.assertGreaterEqual(result.best_confidence, 0.999)


if __name__ == "__main__":
    unittest.main()
