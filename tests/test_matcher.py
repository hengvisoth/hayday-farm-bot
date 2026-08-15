import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
