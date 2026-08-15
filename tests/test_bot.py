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
EMPTY_FIELD_SCREEN = make_screen(4)
PLANT_TOOL_SCREEN = make_screen(5)


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

    def __init__(self):
        self.multiscale_calls = 0

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
            return TemplateMatchResult(
                matches,
                0.95 if matches else 0.1,
                len(matches),
                self.wheat_match,
            )
        if template is bot_module.harvesting_interface_img:
            matches = self.scythe_matches if marker == 3 else []
            return TemplateMatchResult(
                matches,
                0.92 if matches else 0.2,
                len(matches),
                self.scythe_matches[1],
            )
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

    def match_template_multiscale(
        self,
        template,
        target,
        matching_threshold=0.45,
        scales=(1.0,),
        grouping=True,
    ):
        self.multiscale_calls += 1
        result = self.match_template_details(
            template,
            target,
            matching_threshold,
            grouping,
        )
        return TemplateMatchResult(
            result.matches,
            result.best_confidence,
            result.raw_match_count,
            result.best_match,
            1.5,
        )

    def mark_matches(self, matches, target, color):
        return None

    def mark_path(self, points, target):
        return None


class FieldMatcher(FakeMatcher):
    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.field_img:
            return [[100, 100, 20, 20]]
        return super().match_template(template, target, matching_threshold, grouping)


class AdaptivePlantMatcher(FakeMatcher):
    field_matches = [
        [500, 500, 20, 20],
        [550, 500, 20, 20],
        [600, 500, 20, 20],
    ]
    tool_matches = [[900, 900, 30, 30], [560, 430, 30, 30]]

    def match_template_details(
        self,
        template,
        target,
        matching_threshold=0.45,
        grouping=True,
    ):
        marker = int(target[0, 0, 0])
        if (
            template is bot_module.planting_interface_img
            and marker == 5
            and matching_threshold <= 0.7
        ):
            return TemplateMatchResult(
                self.tool_matches,
                0.94,
                len(self.tool_matches),
                self.tool_matches[1],
                1.5,
            )
        if template is bot_module.plant_img or template is bot_module.plant_growing_img:
            return TemplateMatchResult([], 0.1, 0, [100, 100, 20, 20], 1.0)
        return TemplateMatchResult([], 0.1, 0, [100, 100, 20, 20], 1.0)

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        marker = int(target[0, 0, 0])
        if template is bot_module.field_img and marker in (4, 5):
            return self.field_matches
        if template is bot_module.planting_interface_img and marker == 5:
            return self.tool_matches
        return super().match_template(template, target, matching_threshold, grouping)

    def matchs_to_boundary(self, matches, tolerance=50):
        return ((0, 0), (0, 0), (0, 0), (0, 0))

    def boundary_to_path(self, boundary, thickness=55):
        return [(999, 999)]


class LoopPlantMatcher(AdaptivePlantMatcher):
    def __init__(self):
        super().__init__()
        self.wheat_scan_markers = []

    def match_template_details(
        self,
        template,
        target,
        matching_threshold=0.45,
        grouping=True,
    ):
        if template is bot_module.plant_img:
            self.wheat_scan_markers.append(int(target[0, 0, 0]))
        return super().match_template_details(
            template,
            target,
            matching_threshold,
            grouping,
        )


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


class TimedFakeMouse(FakeMouse):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.button_down_at = None
        self.button_hold_seconds = []

    def moveTo(self, x, y, **kwargs):
        super().moveTo(x, y, **kwargs)
        duration = kwargs.get("duration", 0)
        if self.button_down and duration > bot_module.pa.MINIMUM_DURATION:
            self.clock.now += duration

    def mouseDown(self, button="left"):
        super().mouseDown(button)
        self.button_down_at = self.clock.now

    def mouseUp(self, button="left"):
        self.button_hold_seconds.append(self.clock.now - self.button_down_at)
        super().mouseUp(button)


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
    def test_plant_drag_holds_mouse_long_enough_to_register_as_a_gesture(self):
        clock = FakeTime()
        mouse = TimedFakeMouse(clock)
        planting_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=FakeDiagnostics(),
            matcher=FakeMatcher(),
            clock=clock.monotonic,
            waiter=clock.wait,
        )

        result = planting_bot._drag_harvest_route(
            (100, 100),
            [(120, 120), (140, 140)],
            action_state="PLANT",
        )

        self.assertTrue(result)
        self.assertGreaterEqual(mouse.button_hold_seconds[-1], 0.6)
        self.assertFalse(mouse.button_down)

    def test_bot_loop_rescans_after_planting_before_harvest_detection(self):
        capture = SequenceCapture([EMPTY_FIELD_SCREEN, PLANT_TOOL_SCREEN])
        mouse = FakeMouse()
        clock = FakeTime()
        diagnostics = FakeDiagnostics()
        matcher = LoopPlantMatcher()
        planting_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=capture,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=matcher,
            clock=clock.monotonic,
            waiter=clock.wait,
        )
        clock.on_wait = planting_bot.request_stop

        planting_bot.bot_loop()

        self.assertEqual(matcher.wheat_scan_markers.count(4), 1)
        no_wheat_logs = [message for state, message, _ in diagnostics.logs if state == "HARVEST"]
        self.assertNotIn("wheat_not_detected", no_wheat_logs)

    def test_planting_uses_center_field_nearest_tool_and_real_field_route(self):
        capture = SequenceCapture([PLANT_TOOL_SCREEN])
        mouse = FakeMouse()
        clock = FakeTime()
        diagnostics = FakeDiagnostics()
        planting_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=capture,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=AdaptivePlantMatcher(),
            clock=clock.monotonic,
            waiter=clock.wait,
        )

        result = planting_bot.plant_crops(EMPTY_FIELD_SCREEN)

        self.assertTrue(result)
        self.assertEqual(mouse.calls[0], ("click", 550, 500))
        self.assertIn(("moveTo", 560, 430), mouse.calls)
        self.assertIn(("moveTo", 500, 500), mouse.calls)
        self.assertIn(("moveTo", 550, 500), mouse.calls)
        self.assertIn(("moveTo", 600, 500), mouse.calls)
        self.assertEqual(mouse.calls[-1], ("mouseUp", "left"))
        plan_logs = [fields for state, _, fields in diagnostics.logs if state == "PLANT_PLAN"]
        self.assertEqual(plan_logs[-1]["template_scale"], "1.50")
        input_logs = [
            (message, fields)
            for state, message, fields in diagnostics.logs
            if state == "INPUT"
        ]
        self.assertIn(
            ("move_to_plant_tool", {"action_state": "PLANT", "point": (560, 430)}),
            input_logs,
        )

    def test_no_wheat_report_saves_best_candidate_and_match_evidence_once(self):
        harvest_bot, _, _, diagnostics = make_harvest_bot([NO_SCYTHE_SCREEN])
        result = TemplateMatchResult(
            matches=[],
            best_confidence=0.274,
            raw_match_count=3,
            best_match=[620, 410, 55, 35],
        )

        harvest_bot._report_no_wheat(NO_SCYTHE_SCREEN, result)
        harvest_bot._report_no_wheat(NO_SCYTHE_SCREEN, result)

        reports = [fields for state, _, fields in diagnostics.logs if state == "HARVEST"]
        self.assertEqual(reports[-1]["grouped_matches"], 0)
        self.assertEqual(reports[-1]["raw_matches"], 3)
        self.assertEqual(reports[-1]["confidence"], "0.274")
        self.assertEqual(reports[-1]["best_candidate"], (620, 410))
        self.assertEqual(len(diagnostics.failures), 1)
        self.assertEqual(diagnostics.failures[0]["reason"], "wheat_not_detected")
        self.assertEqual(diagnostics.failures[0]["matches"], [[620, 410, 55, 35]])

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
        wheat_logs = [fields for state, _, fields in diagnostics.logs if state == "DETECT_WHEAT"]
        scythe_logs = [fields for state, _, fields in diagnostics.logs if state == "WAIT_FOR_SCYTHE"]
        self.assertEqual(wheat_logs[-1]["raw_matches"], 1)
        self.assertEqual(scythe_logs[-1]["raw_matches"], 2)
        self.assertEqual(wheat_logs[-1]["template_scale"], "1.50")
        self.assertEqual(scythe_logs[-1]["template_scale"], "1.50")
        self.assertGreater(harvest_bot.m.multiscale_calls, 0)

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
