"""Structured logs and failure screenshots for bot diagnostics."""

import re
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2


class Diagnostics:
    def __init__(
        self,
        sink=None,
        directory="diagnostics",
        screenshot_limit=20,
        clock=None,
    ):
        if screenshot_limit < 1:
            raise ValueError("screenshot_limit must be at least one")
        self.sink = sink
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.screenshot_limit = screenshot_limit
        self.clock = clock or datetime.now
        self._lock = Lock()
        self._screenshot_sequence = 0
        started_at = self.clock()
        self.session_path = self.directory / (
            f"session-{started_at:%Y%m%d-%H%M%S-%f}.log"
        )

    def log(self, state, message="", **fields):
        timestamp = self.clock().strftime("%H:%M:%S.%f")[:-3]
        details = " ".join(f"{name}={value}" for name, value in fields.items())
        line = f"[{timestamp}] [{state}] {message} {details}".rstrip()
        with self._lock:
            with self.session_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        if self.sink is not None:
            self.sink(line)
        return line

    def save_failure(
        self,
        reason,
        image,
        matches=(),
        selected=None,
        path=(),
    ):
        marked = image.copy()
        for x, y, width, height in matches:
            top_left = (round(x - width / 2), round(y - height / 2))
            bottom_right = (round(x + width / 2), round(y + height / 2))
            cv2.rectangle(marked, top_left, bottom_right, (255, 0, 0), 2)
            cv2.circle(marked, (int(x), int(y)), 3, (255, 0, 0), -1)

        if selected is not None:
            cv2.circle(marked, tuple(map(int, selected)), 7, (0, 255, 255), 2)

        previous = None
        for point in path:
            current = tuple(map(int, point))
            if previous is not None:
                cv2.line(marked, previous, current, (0, 0, 255), 2)
            cv2.circle(marked, current, 2, (0, 0, 255), -1)
            previous = current

        with self._lock:
            self._screenshot_sequence += 1
            timestamp = self.clock()
            safe_reason = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")
            safe_reason = safe_reason or "failure"
            filename = self.directory / (
                f"failure-{timestamp:%Y%m%d-%H%M%S-%f}-"
                f"{self._screenshot_sequence:04d}-{safe_reason}.png"
            )
            if not cv2.imwrite(str(filename), marked):
                raise OSError(f"Could not write diagnostic screenshot: {filename}")
            self._remove_old_screenshots()
        return filename

    def _remove_old_screenshots(self):
        screenshots = sorted(
            self.directory.glob("failure-*.png"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        for old_file in screenshots[: -self.screenshot_limit]:
            old_file.unlink()
