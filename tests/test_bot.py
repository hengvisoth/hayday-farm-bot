import unittest
from unittest.mock import patch

import numpy as np

import bot as bot_module
from matcher import TemplateMatchResult


def make_screen(marker):
    screen = np.zeros((1080, 1920, 4), dtype=np.uint8)
    screen[0, 0, 0] = marker
    return screen


WHEAT_SCREEN = make_screen(1)
NO_SCYTHE_SCREEN = make_screen(2)
SCYTHE_SCREEN = make_screen(3)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def log(self, *values):
        self.lines.append(" ".join(map(str, values)))


class FakeDiagnostics:
    def __init__(self):
        self.logs = []
        self.failures = []

    def log(self, state, message="", **fields):
        self.logs.append((state, message, fields))

    def save_failure(self, reason, image, matches=(), selected=None, path=()):
        self.failures.append(
            {
                "reason": reason,
                "matches": list(matches),
                "selected": selected,
                "path": list(path),
            }
        )
        return "failure.png"


class FakeMatcher:
    wheat_match = [500, 400, 20, 20]
    scythe_matches = [[900, 900, 30, 30], [560, 350, 30, 30]]

    def match_template_exists(self, template, target, matching_threshold=0.45):
        return False

    def match_template_details(
        self,
        template,
        target,
        matching_threshold=0.45,
        grouping=True,
    ):
        marker = int(target[0, 0, 0])
        if template is bot_module.plant_img:
            matches = [self.wheat_match] if marker in (1, 3) else []
            return TemplateMatchResult(matches, 0.95 if matches else 0.1)
        if template is bot_module.harvesting_interface_img:
            matches = self.scythe_matches if marker == 3 else []
            return TemplateMatchResult(matches, 0.92 if matches else 0.2)
        if template is bot_module.boat_img or template is bot_module.market_img:
            return TemplateMatchResult([], 0.1)
        return TemplateMatchResult([], 0.0)

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        return self.match_template_details(
            template,
            target,
            matching_threshold,
            grouping,
        ).matches

    def mark_matches(self, matches, target, color):
        return None

    def mark_path(self, points, target):
        return None


class FieldMatcher(FakeMatcher):
    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.field_img:
            return [[100, 100, 20, 20]]
        return super().match_template(template, target, matching_threshold, grouping)


class FakeMouse:
    def __init__(self, raise_on_drag=False):
        self.calls = []
        self.button_down = False
        self.raise_on_drag = raise_on_drag

    def click(self, x, y, **kwargs):
        self.calls.append(("click", x, y))

    def moveTo(self, x, y, **kwargs):
        self.calls.append(("moveTo", x, y))
        if self.raise_on_drag and self.button_down:
            raise RuntimeError("drag failed")

    def mouseDown(self, button="left"):
        self.button_down = True
        self.calls.append(("mouseDown", button))

    def mouseUp(self, button="left"):
        self.button_down = False
        self.calls.append(("mouseUp", button))

    def size(self):
        return 1920, 1080


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.on_wait = None

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        if self.on_wait is not None:
            self.on_wait()
        return False


class SequenceCapture:
    def __init__(self, screens):
        self.screens = list(screens)
        self.calls = 0

    def __call__(self):
        screen = self.screens[min(self.calls, len(self.screens) - 1)]
        self.calls += 1
        return screen.copy()


def make_harvest_bot(screens, timeout=4.0, stop_on_wait=False, raise_on_drag=False):
    capture = SequenceCapture(screens)
    mouse = FakeMouse(raise_on_drag=raise_on_drag)
    clock = FakeTime()
    diagnostics = FakeDiagnostics()
    harvest_bot = bot_module.Bot(
        FakeLogger(),
        lambda image: None,
        capture=capture,
        mouse=mouse,
        key_state=lambda key: False,
        diagnostics=diagnostics,
        matcher=FakeMatcher(),
        clock=clock.monotonic,
        waiter=clock.wait,
    )
    harvest_bot.scythe_timeout = timeout
    harvest_bot.scythe_poll_interval = 0.2
    if stop_on_wait:
        clock.on_wait = harvest_bot.request_stop
    return harvest_bot, mouse, capture, diagnostics


class BotHarvestTests(unittest.TestCase):
    def test_planting_does_not_capture_or_click_after_stop(self):
        capture = SequenceCapture([NO_SCYTHE_SCREEN])
        mouse = FakeMouse()
        planting_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=capture,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=FakeDiagnostics(),
            matcher=FieldMatcher(),
        )
        planting_bot.request_stop()

        with patch.object(bot_module.pa, "click", mouse.click):
            planting_bot.plant_crops(NO_SCYTHE_SCREEN)

        self.assertEqual(capture.calls, 0)
        self.assertEqual(mouse.calls, [])

    def test_bot_loop_does_not_capture_when_stop_was_already_requested(self):
        harvest_bot, _, capture, _ = make_harvest_bot([NO_SCYTHE_SCREEN])
        harvest_bot.request_stop()

        with patch.object(bot_module.keyboard, "is_pressed", lambda key: True):
            harvest_bot.bot_loop()

        self.assertEqual(capture.calls, 0)

    def test_harvest_reaches_scythe_without_camera_anchors(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        self.assertEqual(mouse.calls[0], ("click", 500, 400))
        self.assertIn(("moveTo", 560, 350), mouse.calls)
        self.assertIn(("mouseDown", "left"), mouse.calls)
        self.assertEqual(mouse.calls[-1], ("mouseUp", "left"))
        camera_logs = [fields for state, _, fields in diagnostics.logs if state == "CAMERA"]
        self.assertEqual(camera_logs[-1]["method"], "crop_positions")
        self.assertEqual(camera_logs[-1]["shift"], (0, 0))

    def test_harvest_waits_until_scythe_appears(self):
        harvest_bot, mouse, capture, _ = make_harvest_bot(
            [NO_SCYTHE_SCREEN, SCYTHE_SCREEN]
        )

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        self.assertEqual(capture.calls, 2)
        self.assertEqual(
            [call for call in mouse.calls if call[0] == "click"],
            [("click", 500, 400)],
        )

    def test_scythe_timeout_does_not_guess_a_coordinate(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot(
            [NO_SCYTHE_SCREEN],
            timeout=0.4,
        )

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [("click", 500, 400)])
        self.assertEqual(diagnostics.failures[-1]["reason"], "scythe_timeout")

    def test_stop_during_wait_prevents_more_mouse_input(self):
        harvest_bot, mouse, capture, _ = make_harvest_bot(
            [NO_SCYTHE_SCREEN],
            stop_on_wait=True,
        )

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [("click", 500, 400)])
        self.assertEqual(capture.calls, 1)

    def test_mouse_is_released_when_drag_raises(self):
        harvest_bot, mouse, _, _ = make_harvest_bot(
            [SCYTHE_SCREEN],
            raise_on_drag=True,
        )

        with self.assertRaisesRegex(RuntimeError, "drag failed"):
            harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertEqual(mouse.calls[-1], ("mouseUp", "left"))
        self.assertFalse(mouse.button_down)


if __name__ == "__main__":
    unittest.main()
