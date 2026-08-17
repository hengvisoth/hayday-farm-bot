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
AFTER_CROP_SELECTED_SCREEN = make_screen(6)


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
    """Clicking the field is what makes the wheat tool icon appear as a
    real, matchable object, so this fixture models a two-capture flow:
    the field is visible on the initial capture (marker 4), and the
    wheat tool icon only appears on the capture taken after the click
    (marker 5).
    """

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
        if template is bot_module.planting_interface_img and marker == 5:
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
        if template is bot_module.field_img:
            return self.field_matches
        return super().match_template(template, target, matching_threshold, grouping)


class AdaptiveHarvestMatcher(FakeMatcher):
    wheat_matches = [
        [490, 490, 20, 20],
        [550, 490, 20, 20],
        [610, 490, 20, 20],
    ]
    scythe_matches = [[900, 900, 30, 30], [550, 490, 30, 30]]

    def match_template_details(
        self,
        template,
        target,
        matching_threshold=0.45,
        grouping=True,
    ):
        marker = int(target[0, 0, 0])
        if template is bot_module.plant_img:
            matches = self.wheat_matches if marker in (1, 3) else []
            return TemplateMatchResult(
                matches,
                0.95 if matches else 0.1,
                len(matches),
                matches[0] if matches else [100, 100, 20, 20],
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


class FarScytheMatcher(AdaptiveHarvestMatcher):
    """A false-positive scythe match on a fixed HUD icon far from the
    wheat, reproducing the bug where the drag went to (1635, 168) — a
    static corner icon — regardless of which wheat tile was clicked.
    """

    scythe_matches = [[1635, 168, 30, 30], [1600, 150, 30, 30]]


class SelectionMarkerPlantMatcher(AdaptivePlantMatcher):
    field_selected_match = [545, 505, 15, 15]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.field_selected_img:
            return [self.field_selected_match]
        return super().match_template(template, target, matching_threshold, grouping)


class SelectionMarkerHarvestMatcher(AdaptiveHarvestMatcher):
    wheat_selected_match = [545, 495, 15, 15]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.wheat_selected_img:
            return [self.wheat_selected_match]
        return super().match_template(template, target, matching_threshold, grouping)


class ScatteredWheatMatcher(AdaptiveHarvestMatcher):
    """Four matches forming a diamond around an empty gap -- reproduces
    the bug where wheat from separate patches produced a geometric
    center that fell between them, on grass rather than any real tile.
    """

    wheat_matches = [
        [1000, 500, 20, 20],
        [1100, 600, 20, 20],
        [900, 600, 20, 20],
        [1000, 700, 20, 20],
    ]


class StoreMatcher(FakeMatcher):
    store_matches = [[800, 400, 30, 30]]
    store_sold_matches = []
    plant_offer_matches = [[820, 500, 20, 20]]
    new_offer_matches = [[810, 410, 40, 40]]
    newspaper_matches = []
    insert_button_matches = [[830, 600, 60, 20]]

    def match_template_details(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.store_img:
            matches = self.store_matches
        elif template is bot_module.store_sold_img:
            matches = self.store_sold_matches
        elif template is bot_module.new_offer_img:
            matches = self.new_offer_matches
        else:
            return super().match_template_details(template, target, matching_threshold, grouping)
        return TemplateMatchResult(
            matches,
            0.9 if matches else 0.1,
            len(matches),
            matches[0] if matches else [100, 100, 20, 20],
        )

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.plant_offer_img:
            return self.plant_offer_matches
        if template is bot_module.new_offer_img:
            return self.new_offer_matches
        if template is bot_module.newspaper_img:
            return self.newspaper_matches
        if template is bot_module.insert_button_img:
            return self.insert_button_matches
        return super().match_template(template, target, matching_threshold, grouping)


class SellMarketMatcher(FakeMatcher):
    market_matches = [[100, 200, 20, 20]]
    new_offer_matches = [[300, 300, 40, 40], [400, 300, 40, 40]]

    def match_template_details(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.new_offer_img:
            matches = self.new_offer_matches
            return TemplateMatchResult(matches, 0.9, len(matches), matches[0] if matches else None)
        if template is bot_module.market_img:
            matches = self.market_matches
            return TemplateMatchResult(matches, 0.9, len(matches), matches[0] if matches else None)
        return super().match_template_details(template, target, matching_threshold, grouping)

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.market_img:
            return self.market_matches
        if template is bot_module.sold_img or template is bot_module.close_img:
            return []
        return super().match_template(template, target, matching_threshold, grouping)


class StuckStoreMatcher(StoreMatcher):
    """The store opens but never shows a "create new sale" option --
    reproduces a stuck store dialog that should be closed via close.png
    instead of left open indefinitely.
    """

    new_offer_matches = []
    close_matches = [[900, 100, 20, 20]]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.close_img:
            return self.close_matches
        return super().match_template(template, target, matching_threshold, grouping)


class StuckAfterCollectMatcher(StoreMatcher):
    """Collecting money doesn't return to the idle stand (e.g. a reward
    popup opens instead) -- the bot must close it via close.png rather
    than leaving it stuck open for every future loop to trip over.
    """

    store_matches = []
    store_sold_matches = [[850, 300, 25, 25]]
    close_matches = [[900, 100, 20, 20]]

    def match_template_details(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.store_sold_img:
            matches = self.store_sold_matches
            return TemplateMatchResult(matches, 0.9, len(matches), matches[0])
        return super().match_template_details(template, target, matching_threshold, grouping)

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.close_img:
            return self.close_matches
        return super().match_template(template, target, matching_threshold, grouping)


class MultiSlotStoreMatcher(StoreMatcher):
    new_offer_matches = [[810, 410, 40, 40], [900, 410, 40, 40]]


class StoreWithCloseMatcher(StoreMatcher):
    """The stand shows a close button once every offer is created --
    reproduces leaving the shop open after the last insert_button.png
    click instead of exiting it like the market flow does.
    """

    close_matches = [[900, 100, 20, 20]]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        if template is bot_module.close_img:
            return self.close_matches
        return super().match_template(template, target, matching_threshold, grouping)


class NewspaperMatcher(FakeMatcher):
    """The newspaper option only appears on the screen captured *after*
    the crop is selected, not the one from right after opening the sale
    slot -- reproduces a bug where the newspaper check reused that
    earlier, stale screen and could never find it.
    """

    plant_offer_matches = [[400, 400, 20, 20]]
    newspaper_matches = [[450, 450, 20, 20]]
    insert_button_matches = [[500, 500, 30, 10]]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        marker = int(target[0, 0, 0])
        if template is bot_module.plant_offer_img:
            return self.plant_offer_matches
        if template is bot_module.newspaper_img:
            return self.newspaper_matches if marker == 6 else []
        if template is bot_module.insert_button_img:
            return self.insert_button_matches
        return super().match_template(template, target, matching_threshold, grouping)


class DelayedCropMatcher(FakeMatcher):
    """The crop-selection screen doesn't render on the very first check
    after OFFER_OPEN -- reproduces a bug where a single fixed-wait check
    ran too early, found nothing, assumed "plants are empty", and closed
    the whole offer panel via ESC instead of retrying.
    """

    plant_offer_matches = [[400, 400, 20, 20]]
    insert_button_matches = [[500, 500, 30, 10]]

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        marker = int(target[0, 0, 0])
        if template is bot_module.plant_offer_img:
            return self.plant_offer_matches if marker == 2 else []
        if template is bot_module.newspaper_img:
            return []
        if template is bot_module.insert_button_img:
            return self.insert_button_matches
        return super().match_template(template, target, matching_threshold, grouping)


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

    def test_planting_clicks_field_then_holds_wheat_tool_and_drags_through_field(self):
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
        # The field is clicked once to make the wheat tool icon appear,
        # then the tool is held down and dragged through every detected
        # field tile — a rectangular sweep can skip real tiles on a
        # diamond-shaped field, which is exactly what left fields
        # unplanted in live testing.
        self.assertEqual(mouse.calls[0], ("click", 550, 500))
        move_points = [call[1:] for call in mouse.calls if call[0] == "moveTo"]
        self.assertEqual(move_points[0], (560, 430))
        for center in [(500, 500), (550, 500), (600, 500)]:
            self.assertIn(center, move_points)
        self.assertEqual(mouse.calls.count(("mouseDown", "left")), 1)
        self.assertEqual(mouse.calls.count(("mouseUp", "left")), 1)
        plan_logs = [fields for state, _, fields in diagnostics.logs if state == "PLANT_PLAN"]
        self.assertEqual(plan_logs[-1]["template_scale"], "1.50")
        self.assertEqual(plan_logs[-1]["selected"], (560, 430))
        self.assertEqual(plan_logs[-1]["field_matches"], 3)
        input_logs = [
            (message, fields)
            for state, message, fields in diagnostics.logs
            if state == "INPUT"
        ]
        self.assertIn(
            ("move_to_plant_tool", {"action_state": "PLANT", "point": (560, 430)}),
            input_logs,
        )

    def test_planting_row_anchor_prefers_the_selection_marker_when_visible(self):
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
            matcher=SelectionMarkerPlantMatcher(),
            clock=clock.monotonic,
            waiter=clock.wait,
        )

        result = planting_bot.plant_crops(EMPTY_FIELD_SCREEN)

        self.assertTrue(result)
        anchor_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "PLANT" and message == "row_anchor"
        ]
        self.assertTrue(anchor_logs[-1]["selection_marker_found"])
        self.assertEqual(anchor_logs[-1]["row_anchor"], (545, 505))

    def test_planting_returns_false_when_no_empty_fields(self):
        mouse = FakeMouse()
        diagnostics = FakeDiagnostics()
        planting_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=FakeMatcher(),
        )

        result = planting_bot.plant_crops(EMPTY_FIELD_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [])

    def test_plant_tool_timeout_is_reported_as_failure(self):
        capture = SequenceCapture([EMPTY_FIELD_SCREEN])
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
            matcher=FieldMatcher(),
            clock=clock.monotonic,
            waiter=clock.wait,
        )

        result = planting_bot.plant_crops(EMPTY_FIELD_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [("click", 100, 100)])
        self.assertEqual(diagnostics.failures[-1]["reason"], "plant_tool_timeout")

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

    def test_harvest_reaches_scythe_and_drags(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        self.assertEqual(mouse.calls[0], ("click", 500, 400))
        self.assertIn(("moveTo", 560, 350), mouse.calls)
        self.assertIn(("mouseDown", "left"), mouse.calls)
        self.assertEqual(mouse.calls[-1], ("mouseUp", "left"))
        wheat_logs = [fields for state, _, fields in diagnostics.logs if state == "DETECT_WHEAT"]
        scythe_logs = [fields for state, _, fields in diagnostics.logs if state == "WAIT_FOR_SCYTHE"]
        self.assertEqual(wheat_logs[-1]["raw_matches"], 1)
        self.assertEqual(scythe_logs[-1]["raw_matches"], 2)
        self.assertEqual(wheat_logs[-1]["template_scale"], "1.50")
        self.assertEqual(scythe_logs[-1]["template_scale"], "1.50")
        self.assertGreater(harvest_bot.m.multiscale_calls, 0)

    def test_harvest_row_anchor_prefers_the_selection_marker_when_visible(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])
        harvest_bot.m = SelectionMarkerHarvestMatcher()

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        anchor_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "HARVEST" and message == "row_anchor"
        ]
        self.assertTrue(anchor_logs[-1]["selection_marker_found"])
        self.assertEqual(anchor_logs[-1]["row_anchor"], (545, 495))

    def test_harvest_clicks_a_real_wheat_tile_not_the_gap_between_patches(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])
        harvest_bot.m = ScatteredWheatMatcher()

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        # The four matches form a diamond with nothing at its geometric
        # center (1000, 600) -- the click must land on one of the real
        # matched tiles, not that empty midpoint.
        click_calls = [call for call in mouse.calls if call[0] == "click"]
        self.assertEqual(len(click_calls), 1)
        clicked_point = click_calls[0][1:]
        self.assertIn(
            clicked_point,
            [(m[0], m[1]) for m in ScatteredWheatMatcher.wheat_matches],
        )
        self.assertNotEqual(clicked_point, (1000, 600))

    def test_harvest_rejects_a_scythe_match_far_from_the_wheat(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])
        harvest_bot.m = FarScytheMatcher()

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [("click", 550, 490)])
        self.assertEqual(diagnostics.failures[-1]["reason"], "invalid_harvest_plan")

    def test_harvest_visits_every_detected_wheat_tile_with_one_hold(self):
        harvest_bot, mouse, _, diagnostics = make_harvest_bot([SCYTHE_SCREEN])
        harvest_bot.m = AdaptiveHarvestMatcher()

        result = harvest_bot.harvest_plants(WHEAT_SCREEN)

        self.assertTrue(result)
        move_points = [call[1:] for call in mouse.calls if call[0] == "moveTo"]
        # Every detected wheat tile must be visited by the drag path — a
        # rectangular sweep can skip real tiles on a diamond-shaped field.
        for center in [(490, 490), (550, 490), (610, 490)]:
            self.assertIn(center, move_points)
        self.assertEqual(mouse.calls.count(("mouseDown", "left")), 1)
        self.assertEqual(mouse.calls.count(("mouseUp", "left")), 1)
        plan_logs = [fields for state, _, fields in diagnostics.logs if state == "PLAN_DRAG"]
        self.assertEqual(plan_logs[-1]["points"], 9)

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


class BotStoreTests(unittest.TestCase):
    def make_store_bot(self, matcher):
        mouse = FakeMouse()
        clock = FakeTime()
        diagnostics = FakeDiagnostics()
        store_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=lambda: WHEAT_SCREEN,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=matcher,
            clock=clock.monotonic,
            waiter=clock.wait,
        )
        return store_bot, mouse, diagnostics

    def test_read_quantity_near_parses_digits_from_ocr_text(self):
        store_bot, _, _ = self.make_store_bot(FakeMatcher())

        with patch.object(bot_module.pytesseract, "image_to_string", return_value=" 52 \n"):
            quantity, text = store_bot._read_quantity_near(WHEAT_SCREEN, [820, 500, 20, 20])

        self.assertEqual(quantity, 52)
        self.assertEqual(text, "52")

    def test_restock_collects_money_when_sold_marker_visible(self):
        matcher = StoreMatcher()
        matcher.store_sold_matches = [[850, 300, 25, 25]]
        store_bot, mouse, _ = self.make_store_bot(matcher)

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="5"):
            store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertEqual(mouse.calls[0], ("click", 850, 300))

    def test_restock_recovers_via_close_if_stuck_after_collecting_money(self):
        store_bot, mouse, diagnostics = self.make_store_bot(StuckAfterCollectMatcher())

        result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [("click", 850, 300), ("click", 900, 100)])
        recovery_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "RECOVERY" and message == "close_window"
        ]
        self.assertEqual(recovery_logs[-1]["reason"], "store_not_visible_after_collect")

    def test_restock_opens_store_and_creates_a_new_sale_when_quantity_exceeds_threshold(self):
        store_bot, mouse, _ = self.make_store_bot(StoreMatcher())

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="52"):
            result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertTrue(result)
        # Opens the stand, then reuses the same "create a new sale" flow
        # the market uses: open the sale crate, pick the crop, insert it.
        self.assertEqual(
            mouse.calls,
            [
                ("click", 800, 400),
                ("click", 810, 410),
                ("click", 820, 500),
                ("click", 830, 600),
            ],
        )

    def test_restock_creates_every_offer_until_no_slots_remain(self):
        store_bot, mouse, diagnostics = self.make_store_bot(MultiSlotStoreMatcher())
        attempted = []

        def fake_create_offer(offer):
            attempted.append(offer)
            return True

        store_bot.create_offer = fake_create_offer

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="52"):
            result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertTrue(result)
        self.assertEqual(len(attempted), 2)
        complete_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "STORE" and message == "restock_complete"
        ]
        self.assertEqual(complete_logs[-1]["offers_created"], 2)
        self.assertEqual(complete_logs[-1]["slots_seen"], 2)

    def test_restock_closes_store_after_creating_all_offers(self):
        store_bot, mouse, _ = self.make_store_bot(StoreWithCloseMatcher())

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="52"):
            result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertTrue(result)
        # Leaves the shop after the last offer is created instead of
        # leaving it open, mirroring sell_items' "Exit market" step.
        self.assertEqual(mouse.calls[-1], ("click", 900, 100))

    def test_create_offer_checks_newspaper_on_a_fresh_screen_after_crop_selection(self):
        capture = SequenceCapture(
            [WHEAT_SCREEN, AFTER_CROP_SELECTED_SCREEN, AFTER_CROP_SELECTED_SCREEN]
        )
        mouse = FakeMouse()
        diagnostics = FakeDiagnostics()
        offer_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=capture,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=NewspaperMatcher(),
        )

        result = offer_bot.create_offer([300, 300, 40, 40])

        self.assertTrue(result)
        # The newspaper option only exists on the post-selection screen --
        # if the check reused the stale pre-selection screen (marker 1),
        # this click would never happen.
        self.assertIn(("click", 450, 450), mouse.calls)

    def test_create_offer_polls_for_crop_selection_screen_instead_of_one_early_check(self):
        capture = SequenceCapture([WHEAT_SCREEN, NO_SCYTHE_SCREEN])
        mouse = FakeMouse()
        clock = FakeTime()
        diagnostics = FakeDiagnostics()
        offer_bot = bot_module.Bot(
            FakeLogger(),
            lambda image: None,
            capture=capture,
            mouse=mouse,
            key_state=lambda key: False,
            diagnostics=diagnostics,
            matcher=DelayedCropMatcher(),
            clock=clock.monotonic,
            waiter=clock.wait,
        )

        result = offer_bot.create_offer([300, 300, 40, 40])

        self.assertTrue(result)
        # If the old single fixed-wait check were still in place, it would
        # find nothing on the first (too-early) screen, assume "plants
        # are empty", and never click the crop at all.
        self.assertIn(("click", 400, 400), mouse.calls)

    def test_restock_closes_store_if_new_sale_never_appears(self):
        store_bot, mouse, diagnostics = self.make_store_bot(StuckStoreMatcher())

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="52"):
            result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(
            mouse.calls,
            [("click", 800, 400), ("click", 900, 100)],
        )
        timeout_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "STORE" and message == "new_sale_timeout"
        ]
        self.assertEqual(timeout_logs[-1]["seconds"], store_bot.store_timeout)

    def test_restock_skips_when_quantity_at_or_below_threshold(self):
        store_bot, mouse, _ = self.make_store_bot(StoreMatcher())

        with patch.object(bot_module.pytesseract, "image_to_string", return_value="7"):
            result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [])

    def test_restock_does_nothing_when_store_not_visible(self):
        matcher = StoreMatcher()
        matcher.store_matches = []
        store_bot, mouse, _ = self.make_store_bot(matcher)

        result = store_bot.restock_roadside_stand(WHEAT_SCREEN)

        self.assertFalse(result)
        self.assertEqual(mouse.calls, [])

    def test_sell_open_click_scales_offset_with_detected_market_scale(self):
        # The clickable "shop" hotspot isn't a separate template match --
        # it's a fixed offset from the market icon. That offset must scale
        # with the icon's detected size (e.g. the emulator window being a
        # different size) instead of always being a fixed pixel count.
        store_bot, mouse, diagnostics = self.make_store_bot(SellMarketMatcher())

        store_bot.sell_items(WHEAT_SCREEN)

        self.assertEqual(mouse.calls[0], ("click", 175, 200))
        market_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "SELL" and message == "market_found"
        ]
        self.assertEqual(market_logs[-1]["point"], (175, 200))
        self.assertEqual(market_logs[-1]["template_scale"], "1.50")

    def test_sell_items_tries_every_offer_even_if_one_fails(self):
        # A locked "invite a friend" slot (or any other bad match) must
        # not abort every other valid offer -- reproduces a bug where the
        # loop broke after the first failed create_offer call.
        store_bot, mouse, diagnostics = self.make_store_bot(SellMarketMatcher())
        attempted = []

        def fake_create_offer(offer):
            attempted.append(offer)
            return len(attempted) > 1

        store_bot.create_offer = fake_create_offer

        store_bot.sell_items(WHEAT_SCREEN)

        self.assertEqual(len(attempted), 2)
        offer_logs = [
            fields
            for state, message, fields in diagnostics.logs
            if state == "SELL" and message == "new_offers"
        ]
        self.assertEqual(offer_logs[-1]["matches"], 2)


if __name__ == "__main__":
    unittest.main()
