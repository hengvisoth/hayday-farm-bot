import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from diagnostics import Diagnostics


FIXED_TIME = datetime(2026, 8, 16, 14, 22, 10, 125000)


class DiagnosticsTests(unittest.TestCase):
    def test_log_writes_session_file_and_sink(self):
        with tempfile.TemporaryDirectory() as folder:
            emitted = []
            diagnostics = Diagnostics(
                sink=emitted.append,
                directory=folder,
                clock=lambda: FIXED_TIME,
            )

            line = diagnostics.log("HARVEST", "wheat_found", matches=3, confidence="0.93")

            self.assertEqual(
                line,
                "[14:22:10.125] [HARVEST] wheat_found matches=3 confidence=0.93",
            )
            self.assertEqual(emitted, [line])
            self.assertEqual(
                Path(diagnostics.session_path).read_text(encoding="utf-8"),
                line + "\n",
            )

    def test_save_failure_marks_image_and_keeps_latest_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            diagnostics = Diagnostics(
                directory=folder,
                screenshot_limit=2,
                clock=lambda: FIXED_TIME,
            )
            image = np.zeros((30, 30, 3), dtype=np.uint8)

            diagnostics.save_failure("first", image)
            diagnostics.save_failure("second", image)
            latest = diagnostics.save_failure(
                "third failure",
                image,
                matches=[(10, 10, 6, 6)],
                selected=(10, 10),
                path=[(2, 2), (20, 20)],
            )

            screenshots = sorted(Path(folder).glob("failure-*.png"))
            self.assertEqual(len(screenshots), 2)
            self.assertTrue(latest.exists())
            saved = cv2.imread(str(latest), cv2.IMREAD_COLOR)
            self.assertGreater(int(saved.sum()), 0)


if __name__ == "__main__":
    unittest.main()
