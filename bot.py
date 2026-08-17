import os
from threading import Event, Lock
from time import monotonic

from diagnostics import Diagnostics
from matcher import Matcher
from planner import (
    build_field_route,
    select_field_center,
    select_nearest_match,
)

import cv2
import numpy as np
import mss
import keyboard
import pyautogui as pa
import pytesseract

pa.PAUSE = 0

_DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_PATH

# Preparation
FIELD_MATCHING_THRESHOLD = 0.7
HARVEST_MATCHING_THRESHOLD = 0.6
# A tool icon genuinely tied to a click should appear reasonably close to
# it. A "match" far from the click (e.g. a HUD icon in a screen corner
# that coincidentally resembles the template) is rejected instead of
# dragged to, however high its confidence score.
MAX_TOOL_DISTANCE = 500
WHEAT_MATCHING_THRESHOLD = 0.3
WHEAT_GROWING_MATCHING_THRESHOLD = 0.7
BOAT_MATCHING_THRESHOLD = 0.7
MARKET_MATCHING_THRESHOLD = 0.6
SOLD_MATCHING_THRESHOLD = 0.6
NEW_OFFER_MATCHING_THRESHOLD = 0.6
PLANT_OFFER_MATCHING_THRESHOLD = 0.9
PLANTING_TOOL_MATCHING_THRESHOLD = 0.6
FIELD_SELECTED_MATCHING_THRESHOLD = 0.6
WHEAT_SELECTED_MATCHING_THRESHOLD = 0.6
NEWSPAPER_MATCHING_THRESHOLD = 0.6
INSERT_BUTTON_MATCHING_THRESHOLD = 0.9
SILO_MATCHING_THRESHOLD = 0.9
CLOSE_MATCHING_THRESHOLD = 0.4
STORE_MATCHING_THRESHOLD = 0.6
# Observed false positives on store_sold.png cluster tightly at 0.618-0.625
# confidence (a coincidentally similar-looking world object, not the real
# "SOLD!" banner, which has never yet matched at all). Set with a solid
# margin above that band; revisit once a genuine sale is ever observed.
STORE_SOLD_MATCHING_THRESHOLD = 0.75
STORE_RESTOCK_MIN_QUANTITY = 10
DETECTION_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

_detected_width, _detected_height = pa.size()
SCREEN_DIM = {
    'left': 0,
    'top': 0,
    'width': _detected_width,
    'height': _detected_height,
}

BOAT_ANCHOR = (1075, 285)
MARKET_ANCHOR = (770, 1150)

def _load_template(path):
    """Load a template and normalize it to 4-channel BGRA.

    mss screen captures are always BGRA. cv2.matchTemplate requires the
    template and target to share a channel count, so a template
    re-saved by an image editor without an alpha channel (3-channel
    BGR) would otherwise crash matchTemplate the moment it's used.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


plant_img = _load_template('templates/plants/wheat.png')
plant_growing_img = _load_template('templates/plants/wheat_growing.png')
planting_interface_img = _load_template('templates/interface/planting_wheat.png')
field_selected_img = _load_template('templates/interface/field_selected.png')
wheat_selected_img = _load_template('templates/interface/wheat_selected.png')
field_img = _load_template('templates/environment/field.png')
harvesting_interface_img = _load_template('templates/interface/harvest_scythe.png')
boat_img = _load_template('templates/environment/boat.png')
market_img = _load_template('templates/environment/market.png')
sold_img = _load_template('templates/interface/sold.png')
new_offer_img = _load_template('templates/interface/new_offer.png')
plant_offer_img = _load_template('templates/interface/wheat_market.png')
newspaper_img = _load_template('templates/interface/newspaper.png')
insert_button_img = _load_template('templates/interface/insert_button.png')
silo_img = _load_template('templates/interface/silo.png')
close_img = _load_template('templates/interface/close.png')
store_img = _load_template('templates/environment/store.png')
store_sold_img = _load_template('templates/environment/store_sold.png')


class Bot:
    def __init__(
        self,
        logger,
        set_tracking_img,
        capture=None,
        mouse=None,
        key_state=None,
        diagnostics=None,
        matcher=None,
        clock=None,
        waiter=None,
    ):
        self.logger = logger
        self.set_tracking_img = set_tracking_img
        self.m = matcher or Matcher()
        self.mouse = mouse or pa
        self.key_state = key_state or keyboard.is_pressed
        self.clock = clock or monotonic
        self.stop_event = Event()
        self.waiter = waiter or self.stop_event.wait
        self.capture = capture or self._capture_screen
        sink = getattr(logger, "log", logger if callable(logger) else None)
        self.diagnostics = diagnostics or Diagnostics(sink=sink)
        self._loop_lock = Lock()
        self.scythe_timeout = 4.0
        self.scythe_poll_interval = 0.2
        self.store_timeout = 5.0
        self.drag_press_hold = 0.25
        self.drag_point_duration = 0.15
        self.drag_point_settle = 0.05
        self.drag_release_hold = 0.1
        self._wheat_scan_saved = False
        self.silo_is_full = False
        self.harvested_plants = False
        self.planted_crops = 0

    def _capture_screen(self):
        with mss.MSS() as sct:
            return np.array(sct.grab(SCREEN_DIM))

    def get_target(self):
        return self.capture()

    def request_stop(self):
        self.stop_event.set()
        self._log("CONTROL", "stop_requested")

    def reset_stop(self):
        self.stop_event.clear()

    def _stop_requested(self):
        if self.stop_event.is_set():
            return True
        try:
            if self.key_state("q"):
                self.stop_event.set()
                return True
        except Exception as error:
            self._log("CONTROL", "key_check_failed", error=repr(error))
        return False

    def _wait(self, seconds):
        if self._stop_requested():
            return True
        stopped = bool(self.waiter(max(0, seconds)))
        return stopped or self._stop_requested()

    def _log(self, state, message="", **fields):
        return self.diagnostics.log(state, message, **fields)

    def get_anchor_positions(self, target):
        anchors = {}
        boat = self.m.match_template(boat_img, target, BOAT_MATCHING_THRESHOLD)
        if boat:
            anchors["boat"] = (boat[0][0], boat[0][1])
        market = self.m.match_template(market_img, target, MARKET_MATCHING_THRESHOLD)
        if market:
            anchors["market"] = (market[0][0], market[0][1])
        return anchors

    def get_camera_pos(self, target=None):
        target = self.get_target() if target is None else target
        boat = self.m.match_template(boat_img, target, BOAT_MATCHING_THRESHOLD)
        if len(boat) > 0:
            ax, ay = BOAT_ANCHOR
            # self.logger.log("Target: BOAT_ANCHOR", ax, ay, boat[0][0], boat[0][1])
            return ax + ax - boat[0][0], ay + ay - boat[0][1]

        market = self.m.match_template(market_img, target, MARKET_MATCHING_THRESHOLD)
        if len(market) > 0:
            ax, ay = MARKET_ANCHOR
            dx, dy = ax - BOAT_ANCHOR[0], ay - BOAT_ANCHOR[1]
            # self.logger.log("Target: MARKET_ANCHOR", ax, ay, market[0][0], market[0][1])
            return ax + ax - market[0][0] - dx, ay + ay - market[0][1] - dy

        return 0, 0

    def check_fields_are_empty(self, target):
        return self.m.match_template_exists(field_img, target, FIELD_MATCHING_THRESHOLD)

    def check_plants_are_growing(self, target):
        result = self._match_details(
            plant_growing_img,
            target,
            WHEAT_GROWING_MATCHING_THRESHOLD,
        )
        return bool(result.matches)

    def check_silo_is_full(self, target):
        return len(self.m.match_template(silo_img, target, SILO_MATCHING_THRESHOLD)) > 0

    def _read_quantity_near(self, target, match, offset=(-40, 10), size=(80, 30)):
        """OCR a small region near a matched icon to read a quantity badge.

        The offset/size are a starting guess for where the game draws the
        quantity text relative to the icon -- tune them against a real
        screenshot once the raw OCR text is visible in the logs.
        """
        x, y, w, h = match[0], match[1], match[2], match[3]
        dx, dy = offset
        box_w, box_h = size
        left = max(0, x + dx)
        top = max(0, y + h // 2 + dy)
        crop = target[top:top + box_h, left:left + box_w]
        if crop.size == 0:
            return None, ""
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGRA2GRAY if crop.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
        )
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            scaled,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        ).strip()
        return (int(text) if text.isdigit() else None), text

    def restock_roadside_stand(self, target):
        store_sold_result = self._match_details(store_sold_img, target, STORE_SOLD_MATCHING_THRESHOLD)
        collected = False
        if store_sold_result.matches:
            point = (store_sold_result.matches[0][0], store_sold_result.matches[0][1])
            self._log(
                "STORE",
                "collect_money",
                point=point,
                confidence=f"{store_sold_result.best_confidence:.3f}",
                template_scale=f"{store_sold_result.template_scale:.2f}",
            )
            if not self._click(point, "STORE_COLLECT"):
                return False
            collected = True
            if self._wait(1.0):
                return False
            target = self.get_target()

        store_result = self._match_details(store_img, target, STORE_MATCHING_THRESHOLD)
        if not store_result.matches:
            self._log(
                "STORE",
                "not_visible",
                confidence=f"{store_result.best_confidence:.3f}",
                template_scale=f"{store_result.template_scale:.2f}",
                best_candidate=None
                if store_result.best_match is None
                else tuple(store_result.best_match[:2]),
            )
            # Collecting money may have opened something we don't
            # recognize (a reward popup, etc.) instead of returning to
            # the idle stand -- try to close it instead of leaving it
            # stuck open for every future loop iteration to trip over.
            if collected:
                self.check_unexpected_behaviour(target, reason="store_not_visible_after_collect")
            return False
        store = store_result.matches

        wheat_matches = self.m.match_template(plant_offer_img, target, PLANT_OFFER_MATCHING_THRESHOLD)
        if not wheat_matches:
            self._log("STORE", "wheat_quantity_not_visible")
            return False

        quantity, raw_text = self._read_quantity_near(target, wheat_matches[0])
        self._log("STORE", "wheat_quantity", quantity=quantity, raw_text=raw_text)
        if quantity is None or quantity <= STORE_RESTOCK_MIN_QUANTITY:
            return False

        point = (store[0][0], store[0][1])
        self._log("STORE", "restock_open", point=point, quantity=quantity)
        if not self._click(point, "STORE_OPEN"):
            return False

        # Confirm the stand actually opened by waiting for the first
        # "create new sale" slot to appear.
        new_sale_target, first_pass_matches = self._wait_for_new_sale()
        if not first_pass_matches:
            self._log("STORE", "new_sale_timeout", seconds=self.store_timeout)
            if new_sale_target is not None:
                self.check_unexpected_behaviour(new_sale_target, reason="new_sale_timeout")
            return False

        # Re-detect new_offer_img fresh before every click and repeat
        # until no slots remain, instead of filling just one -- bounded by
        # the first pass's count so a persistent bad match (e.g. a locked
        # slot) can't loop forever.
        max_offers = len(first_pass_matches)
        offers_created = 0
        for _ in range(max_offers):
            if self._stop_requested():
                break
            offer_target = self.get_target()
            new_sale_result = self._match_details(new_offer_img, offer_target, NEW_OFFER_MATCHING_THRESHOLD)
            new_sale_matches = new_sale_result.matches
            self._log(
                "STORE",
                "new_sale",
                matches=len(new_sale_matches),
                raw_matches=new_sale_result.raw_match_count,
                confidence=f"{new_sale_result.best_confidence:.3f}",
                template_scale=f"{new_sale_result.template_scale:.2f}",
            )
            if not new_sale_matches:
                break
            if self.create_offer(new_sale_matches[0]):
                offers_created += 1

        self._log("STORE", "restock_complete", offers_created=offers_created, slots_seen=max_offers)

        # Leave the shop instead of leaving it open for the next loop
        # iteration to trip over -- mirrors sell_items' market exit.
        exit_target = self.get_target()
        close = self.m.match_template(close_img, exit_target, CLOSE_MATCHING_THRESHOLD)
        if close:
            point = (close[0][0], close[0][1])
            self._log("STORE", "leave_shop", point=point)
            self._click(point, "STORE_LEAVE")

        return offers_created > 0

    def _wait_for_new_sale(self):
        deadline = self.clock() + self.store_timeout
        last_target = None
        while not self._stop_requested():
            last_target = self.get_target()
            matches = self.m.match_template(new_offer_img, last_target, NEW_OFFER_MATCHING_THRESHOLD)
            if matches:
                return last_target, matches
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if self._wait(min(self.scythe_poll_interval, remaining)):
                return last_target, None
        return last_target, None

    def drag_operation(self, drag_start, matches, target):
        boundary = self.m.matchs_to_boundary(matches)
        path = self.m.boundary_to_path(boundary)
        self.track_path(path, target)
        if self._stop_requested():
            return False
        self.mouse.moveTo(drag_start[0], drag_start[1])
        if self._stop_requested():
            return False
        self.mouse.mouseDown(button='left')
        try:
            for x, y in path:
                if self._stop_requested():
                    return False
                self.mouse.moveTo(x, y, duration=0.1, _pause=False)
            return True
        finally:
            self.mouse.mouseUp(button='left')

    def combine_paths(self, p1, p2):
        result = []
        for (x, y, w, h) in p1:
            result.append([x, y, w, h])
        for (x, y, w, h) in p2:
            result.append([x, y, w, h])
        return result

    def translate_path(self, path, translation):
        result = []
        tx, ty = translation
        for (x, y, w, h) in path:
            result.append([x + tx, y + ty, w, h])
        return result

    def plant_crops(self, target):
        if self._stop_requested():
            return False
        grown_result = self._match_details(
            plant_img,
            target,
            WHEAT_MATCHING_THRESHOLD,
        )
        if grown_result.matches:
            self._log("PLANT", "grown_wheat_present")
            return False

        empty_fields = self.m.match_template(field_img, target, FIELD_MATCHING_THRESHOLD)
        if not empty_fields:
            self._log("PLANT", "empty_fields_gone")
            return False

        field_center = select_field_center(empty_fields)
        # Click an actual matched field tile, not the group's geometric
        # center -- if the matches come from several separate patches
        # (not one contiguous block), the center can fall in the gap
        # between them, landing the click on grass instead of a tile.
        click_match = select_nearest_match(empty_fields, field_center)
        click_point = (click_match[0], click_match[1])
        self._log(
            "PLANT",
            "select_field",
            matches=len(empty_fields),
            selected=click_point,
        )
        if not self._click(click_point, "PLANT_SELECT"):
            return False

        wheat_target, wheat_result = self._wait_for_plant_tool()
        if wheat_result is None:
            if not self._stop_requested() and wheat_target is not None:
                screenshot = self.diagnostics.save_failure(
                    "plant_tool_timeout",
                    wheat_target,
                    matches=empty_fields,
                    selected=field_center,
                )
                self._log(
                    "WAIT_FOR_PLANT_TOOL",
                    "timeout",
                    seconds=self.scythe_timeout,
                    screenshot=screenshot,
                )
            return False

        # Re-detect the field fresh on the post-click screen instead of
        # reusing the pre-click positions -- stays correct even if the
        # click (or anything else) moved the camera.
        current_fields = self.m.match_template(field_img, wheat_target, FIELD_MATCHING_THRESHOLD)
        current_center = select_field_center(current_fields)
        if current_center is None:
            screenshot = self.diagnostics.save_failure(
                "invalid_plant_plan",
                wheat_target,
                matches=wheat_result.matches,
                selected=field_center,
            )
            self._log("PLANT_PLAN", "invalid", screenshot=screenshot)
            return False

        tool_match = select_nearest_match(
            wheat_result.matches,
            current_center,
            max_distance=MAX_TOOL_DISTANCE,
        )
        if tool_match is None:
            screenshot = self.diagnostics.save_failure(
                "invalid_plant_plan",
                wheat_target,
                matches=wheat_result.matches,
                selected=current_center,
            )
            self._log("PLANT_PLAN", "invalid", screenshot=screenshot)
            return False

        tool = (tool_match[0], tool_match[1])
        selected_field_result = self.m.match_template(
            field_selected_img,
            wheat_target,
            FIELD_SELECTED_MATCHING_THRESHOLD,
        )
        selected_field_match = select_nearest_match(
            selected_field_result,
            field_center,
            max_distance=MAX_TOOL_DISTANCE,
        )
        if selected_field_match is not None:
            row_anchor = (selected_field_match[0], selected_field_match[1])
        else:
            row_anchor_match = select_nearest_match(current_fields, current_center)
            row_anchor = (row_anchor_match[0], row_anchor_match[1])
        self._log(
            "PLANT",
            "row_anchor",
            selection_marker_found=selected_field_match is not None,
            row_anchor=row_anchor,
        )
        route = build_field_route(current_fields, tool, row_start=row_anchor, max_segment=25)
        self._log(
            "PLANT_PLAN",
            "complete",
            field_matches=len(current_fields),
            tool_matches=len(wheat_result.matches),
            raw_matches=wheat_result.raw_match_count,
            confidence=f"{wheat_result.best_confidence:.3f}",
            template_scale=f"{wheat_result.template_scale:.2f}",
            selected=tool,
            points=len(route),
        )
        self._track_harvest_plan(
            wheat_target,
            wheat_result.matches,
            current_center,
            tool,
            route,
        )
        return self._drag_harvest_route(tool, route, action_state="PLANT")

    def _match_details(self, template, target, threshold):
        return self.m.match_template_multiscale(
            template,
            target,
            threshold,
            scales=DETECTION_SCALES,
        )

    @staticmethod
    def _match_centers(matches):
        return [(match[0], match[1]) for match in matches]

    def _report_no_wheat(self, target, result):
        best_candidate = None
        best_matches = []
        if result.best_match is not None:
            best_candidate = (result.best_match[0], result.best_match[1])
            best_matches = [result.best_match]

        screenshot = None
        if not self._wheat_scan_saved:
            screenshot = self.diagnostics.save_failure(
                "wheat_not_detected",
                target,
                matches=best_matches,
                selected=best_candidate,
            )
            self._wheat_scan_saved = True

        self._log(
            "HARVEST",
            "wheat_not_detected",
            grouped_matches=len(result.matches),
            raw_matches=result.raw_match_count,
            confidence=f"{result.best_confidence:.3f}",
            template_scale=f"{result.template_scale:.2f}",
            threshold=WHEAT_MATCHING_THRESHOLD,
            best_candidate=best_candidate,
            screenshot=screenshot,
        )

    def _click(self, point, state, **kwargs):
        if self._stop_requested():
            return False
        self._log("INPUT", "click", action_state=state, point=point, **kwargs)
        self.mouse.click(point[0], point[1], **kwargs)
        return True

    def _wait_for_scythe(self):
        deadline = self.clock() + self.scythe_timeout
        last_target = None
        while not self._stop_requested():
            last_target = self.get_target()
            result = self._match_details(
                harvesting_interface_img,
                last_target,
                HARVEST_MATCHING_THRESHOLD,
            )
            self._log(
                "WAIT_FOR_SCYTHE",
                "poll",
                matches=len(result.matches),
                raw_matches=result.raw_match_count,
                confidence=f"{result.best_confidence:.3f}",
                template_scale=f"{result.template_scale:.2f}",
                best_candidate=None
                if result.best_match is None
                else tuple(result.best_match[:2]),
                candidates=self._match_centers(result.matches),
            )
            if result.matches:
                return last_target, result
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if self._wait(min(self.scythe_poll_interval, remaining)):
                return None, None
        return last_target, None

    def _wait_for_plant_tool(self):
        deadline = self.clock() + self.scythe_timeout
        last_target = None
        while not self._stop_requested():
            last_target = self.get_target()
            result = self._match_details(
                planting_interface_img,
                last_target,
                PLANTING_TOOL_MATCHING_THRESHOLD,
            )
            self._log(
                "WAIT_FOR_PLANT_TOOL",
                "poll",
                matches=len(result.matches),
                raw_matches=result.raw_match_count,
                confidence=f"{result.best_confidence:.3f}",
                template_scale=f"{result.template_scale:.2f}",
                best_candidate=None
                if result.best_match is None
                else tuple(result.best_match[:2]),
            )
            if result.matches:
                return last_target, result
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if self._wait(min(self.scythe_poll_interval, remaining)):
                return None, None
        return last_target, None

    def _drag_harvest_route(self, scythe, route, action_state="HARVEST"):
        if self._stop_requested():
            return False
        move_message = (
            "move_to_plant_tool"
            if action_state == "PLANT"
            else "move_to_scythe"
        )
        self._log(
            "INPUT",
            move_message,
            action_state=action_state,
            point=scythe,
        )
        self.mouse.moveTo(scythe[0], scythe[1])
        if self._stop_requested():
            return False

        mouse_is_down = False
        mouse_down_at = None
        try:
            self._log(
                "INPUT",
                "mouse_down",
                action_state=action_state,
                point=scythe,
                press_hold=self.drag_press_hold,
                point_duration=self.drag_point_duration,
                point_settle=self.drag_point_settle,
            )
            self.mouse.mouseDown(button="left")
            mouse_is_down = True
            mouse_down_at = self.clock()
            if self._wait(self.drag_press_hold):
                return False
            for point in route:
                if self._stop_requested():
                    return False
                self.mouse.moveTo(
                    point[0],
                    point[1],
                    duration=self.drag_point_duration,
                    _pause=False,
                )
                if self._wait(self.drag_point_settle):
                    return False
            if self._wait(self.drag_release_hold):
                return False
            return True
        finally:
            if mouse_is_down:
                self.mouse.mouseUp(button="left")
                held_seconds = self.clock() - mouse_down_at
                self._log(
                    "INPUT",
                    "mouse_up",
                    action_state=action_state,
                    held_seconds=f"{held_seconds:.3f}",
                )

    def _track_harvest_plan(self, target, matches, selected, scythe, path):
        tracked = target.copy()
        self.m.mark_matches(matches, tracked, (255, 0, 0))
        cv2.circle(tracked, selected, 7, (0, 255, 255), 2)
        cv2.circle(tracked, scythe, 7, (255, 255, 0), 2)
        self.m.mark_path([scythe, *path], tracked)
        self.set_tracking_img(tracked)

    def harvest_plants(self, target):
        if self._stop_requested():
            return False
        if self.check_plants_are_growing(target):
            self._log("DETECT_WHEAT", "plants_still_growing")
            return False

        wheat_result = self._match_details(
            plant_img,
            target,
            WHEAT_MATCHING_THRESHOLD,
        )
        crop_center = select_field_center(wheat_result.matches)
        # Click an actual matched wheat tile, not the group's geometric
        # center -- if the matches come from several separate patches
        # (not one contiguous block), the center can fall in the gap
        # between them, landing the click on grass instead of wheat.
        selected_match = select_nearest_match(wheat_result.matches, crop_center) if crop_center else None
        selected_crop = (selected_match[0], selected_match[1]) if selected_match else None
        self._log(
            "DETECT_WHEAT",
            "complete",
            matches=len(wheat_result.matches),
            raw_matches=wheat_result.raw_match_count,
            confidence=f"{wheat_result.best_confidence:.3f}",
            template_scale=f"{wheat_result.template_scale:.2f}",
            best_candidate=None
            if wheat_result.best_match is None
            else tuple(wheat_result.best_match[:2]),
            candidates=self._match_centers(wheat_result.matches),
            selected=selected_crop,
        )
        if selected_crop is None:
            return False

        self._log("OPEN_TOOL", "click_wheat", point=selected_crop)
        if not self._click(selected_crop, "OPEN_TOOL"):
            return False

        scythe_target, scythe_result = self._wait_for_scythe()
        if scythe_result is None:
            if not self._stop_requested() and scythe_target is not None:
                screenshot = self.diagnostics.save_failure(
                    "scythe_timeout",
                    scythe_target,
                    matches=wheat_result.matches,
                    selected=selected_crop,
                )
                self._log(
                    "WAIT_FOR_SCYTHE",
                    "timeout",
                    seconds=self.scythe_timeout,
                    screenshot=screenshot,
                )
            return False

        # Re-detect wheat fresh on the post-click screen instead of
        # translating the pre-click positions through an estimated camera
        # shift -- stays correct even if the click (or anything else)
        # moved the camera.
        current_wheat = self._match_details(
            plant_img,
            scythe_target,
            WHEAT_MATCHING_THRESHOLD,
        )
        current_center = select_field_center(current_wheat.matches)
        if current_center is None:
            screenshot = self.diagnostics.save_failure(
                "invalid_harvest_plan",
                scythe_target,
                matches=scythe_result.matches,
                selected=selected_crop,
            )
            self._log("PLAN_DRAG", "invalid", screenshot=screenshot)
            return False

        selected_scythe_match = select_nearest_match(
            scythe_result.matches,
            current_center,
            max_distance=MAX_TOOL_DISTANCE,
        )
        if selected_scythe_match is None:
            screenshot = self.diagnostics.save_failure(
                "invalid_harvest_plan",
                scythe_target,
                matches=scythe_result.matches,
                selected=current_center,
            )
            self._log("PLAN_DRAG", "invalid", screenshot=screenshot)
            return False

        scythe = (selected_scythe_match[0], selected_scythe_match[1])
        selected_wheat_result = self.m.match_template(
            wheat_selected_img,
            scythe_target,
            WHEAT_SELECTED_MATCHING_THRESHOLD,
        )
        selected_wheat_match = select_nearest_match(
            selected_wheat_result,
            selected_crop,
            max_distance=MAX_TOOL_DISTANCE,
        )
        if selected_wheat_match is not None:
            row_anchor = (selected_wheat_match[0], selected_wheat_match[1])
        else:
            row_anchor_match = select_nearest_match(current_wheat.matches, current_center)
            row_anchor = (row_anchor_match[0], row_anchor_match[1])
        self._log(
            "HARVEST",
            "row_anchor",
            selection_marker_found=selected_wheat_match is not None,
            row_anchor=row_anchor,
        )
        route = build_field_route(current_wheat.matches, scythe, row_start=row_anchor, max_segment=25)
        self._log(
            "PLAN_DRAG",
            "complete",
            scythe_matches=len(scythe_result.matches),
            raw_matches=scythe_result.raw_match_count,
            confidence=f"{scythe_result.best_confidence:.3f}",
            template_scale=f"{scythe_result.template_scale:.2f}",
            candidates=self._match_centers(scythe_result.matches),
            selected=scythe,
            points=len(route),
            start=scythe,
            end=route[-1],
        )
        self._track_harvest_plan(
            scythe_target,
            scythe_result.matches,
            current_center,
            scythe,
            route,
        )
        if not self._drag_harvest_route(scythe, route):
            return False

        self.harvested_plants = True
        self._log("DRAG", "complete", points=len(route))
        return True

    def sell_items(self, target):
        if self._stop_requested():
            return False
        self._log("SELL", "started")
        # Open market
        market_result = self._match_details(market_img, target, MARKET_MATCHING_THRESHOLD)
        if not market_result.matches:
            self._log(
                "SELL",
                "market_not_found",
                confidence=f"{market_result.best_confidence:.3f}",
                template_scale=f"{market_result.template_scale:.2f}",
            )
            return False
        market = market_result.matches
        # The clickable "shop" hotspot is offset from the detected market
        # icon rather than a separate template match -- scale that offset
        # by the same factor the icon itself matched at, so it still lands
        # correctly if the game is rendering at a different size (e.g. the
        # emulator window was resized) instead of a fixed pixel count.
        sell_open_point = (
            round(market[0][0] + 50 * market_result.template_scale),
            market[0][1],
        )
        self._log(
            "SELL",
            "market_found",
            point=sell_open_point,
            confidence=f"{market_result.best_confidence:.3f}",
            template_scale=f"{market_result.template_scale:.2f}",
        )
        if self._wait(0.2):
            return False
        if not self._click(sell_open_point, "SELL_OPEN"):
            return False
        if self._wait(1.0):
            return False

        # Collect coins
        target = self.get_target()
        sold = self.m.match_template(sold_img, target, SOLD_MATCHING_THRESHOLD)
        self.track_matches(sold, target)
        if len(sold) > 0:
            self.logger.log("Collecting coins...")
        for s in sold:
            if self._wait(0.2) or not self._click((s[0], s[1]), "SELL_COLLECT"):
                return False
        if self._wait(1.0):
            return False

        # Create offers. Re-detect new_offer_img fresh before every single
        # click instead of clicking through one stale list of coordinates
        # -- filling a slot (or anything else) shifting the panel would
        # otherwise send later clicks to the wrong place. Bounded by the
        # first pass's match count so a persistent bad match (e.g. a
        # locked "invite a friend" slot) can't loop forever.
        target = self.get_target()
        first_pass = self._match_details(new_offer_img, target, NEW_OFFER_MATCHING_THRESHOLD)
        max_offers = len(first_pass.matches)
        if max_offers > 0:
            self.logger.log("Inserting new offers...")
        else:
            self.logger.log("No slots for offers found!")

        for _ in range(max_offers):
            if self._stop_requested():
                break
            target = self.get_target()
            new_offer_result = self._match_details(new_offer_img, target, NEW_OFFER_MATCHING_THRESHOLD)
            new_offers = new_offer_result.matches
            self.track_matches(new_offers, target)
            self._log(
                "SELL",
                "new_offers",
                matches=len(new_offers),
                raw_matches=new_offer_result.raw_match_count,
                confidence=f"{new_offer_result.best_confidence:.3f}",
                template_scale=f"{new_offer_result.template_scale:.2f}",
                candidates=self._match_centers(new_offers),
            )
            if not new_offers:
                break
            # Don't let one bad slot (e.g. that locked slot) abort the
            # rest -- keep going regardless of this attempt's outcome.
            self.create_offer(new_offers[0])
        if self._wait(1.0):
            return False

        # Exit market
        close = self.m.match_template(close_img, target, CLOSE_MATCHING_THRESHOLD)
        if len(close) > 0:
            self.logger.log("Finished with selling. Closing market...")
            return self._click((close[0][0], close[0][1]), "SELL_CLOSE")
        return True

    def _wait_for_match(self, template, threshold):
        """Poll for a template instead of a single check after a fixed
        wait -- a UI transition that takes longer than the fixed wait
        would otherwise make the check run too early and find nothing
        every single time, not just occasionally.
        """
        deadline = self.clock() + self.scythe_timeout
        last_target = None
        while not self._stop_requested():
            last_target = self.get_target()
            matches = self.m.match_template(template, last_target, threshold)
            if matches:
                return last_target, matches
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if self._wait(min(self.scythe_poll_interval, remaining)):
                return last_target, None
        return last_target, None

    def create_offer(self, offer):
        # Open new offer window
        if not self._click((offer[0], offer[1]), "OFFER_OPEN"):
            return False

        # Select target plant to sell. Poll instead of a single fixed-wait
        # check -- the crop-selection screen can take longer to render
        # than a flat 0.5s, and a too-early check found nothing every
        # time, assumed "plants are empty", and closed the whole panel.
        target, plant_to_sell = self._wait_for_match(plant_offer_img, PLANT_OFFER_MATCHING_THRESHOLD)
        if plant_to_sell:
            self.track_matches(plant_to_sell, target)
            if not self._click(
                (plant_to_sell[0][0], plant_to_sell[0][1]),
                "OFFER_SELECT_CROP",
            ):
                return False
        else:
            self.logger.log("Target plant not found... Assuming, that plants are empty")
            if not self._stop_requested():
                keyboard.send("esc")
            return False
        if self._wait(0.5):
            return False

        # Check if newspaper insert is available. Re-capture fresh here --
        # this used to reuse the screen from before the crop was even
        # selected, so the newspaper option (which only appears after
        # that click) was checked against a stale, pre-selection screen.
        target = self.get_target()
        newspaper = self.m.match_template(newspaper_img, target, NEWSPAPER_MATCHING_THRESHOLD)
        self.track_matches(newspaper, target)
        if len(newspaper) > 0:
            self.logger.log("Insert to newspaper available. Inserting...")
            if not self._click(
                (newspaper[0][0], newspaper[0][1]),
                "OFFER_NEWSPAPER",
            ):
                return False
        if self._wait(0.2):
            return False

        # Insert offer
        target = self.get_target()
        insert_button = self.m.match_template(insert_button_img, target, INSERT_BUTTON_MATCHING_THRESHOLD)
        self.track_matches(insert_button, target)
        if len(insert_button) > 0:
            if not self._click(
                (insert_button[0][0], insert_button[0][1]),
                "OFFER_INSERT",
            ):
                return False
        else:
            self.logger.log("Insert button not found. This is a critical error, because no plants can be sold.")
            return False
        return not self._wait(0.2)

    def check_unexpected_behaviour(self, target, reason=None):
        if reason is None:
            self._log("RECOVERY", "skipped_without_known_failure")
            return False
        close = self.m.match_template(close_img, target, CLOSE_MATCHING_THRESHOLD)
        if len(close) > 0:
            self._log("RECOVERY", "close_window", reason=reason)
            if self._wait(0.2):
                return False
            return self._click((close[0][0], close[0][1]), "RECOVERY_CLOSE")
        return False

    def track_matches(self, matches, target):
        tracked = target.copy()
        self.m.mark_matches(matches, tracked, (255, 0, 0))
        self.set_tracking_img(tracked)

    def track_path(self, path, target):
        tracked = target.copy()
        self.m.mark_path(path, tracked)
        self.set_tracking_img(tracked)

    def bot_loop(self):
        if not self._loop_lock.acquire(blocking=False):
            self._log("CONTROL", "worker_already_running")
            return False

        try:
            try:
                mouse_size = tuple(self.mouse.size())
            except Exception as error:
                mouse_size = f"unavailable:{error!r}"
            templates_ok = all(
                image is not None
                for image in (
                    plant_img,
                    plant_growing_img,
                    planting_interface_img,
                    field_selected_img,
                    wheat_selected_img,
                    field_img,
                    harvesting_interface_img,
                    boat_img,
                    market_img,
                    store_img,
                    store_sold_img,
                )
            )
            self._log(
                "SESSION",
                "started",
                capture_bounds=SCREEN_DIM,
                mouse_size=mouse_size,
                templates_ok=templates_ok,
                wheat_threshold=WHEAT_MATCHING_THRESHOLD,
                scythe_threshold=HARVEST_MATCHING_THRESHOLD,
                scythe_timeout=self.scythe_timeout,
            )

            while not self._stop_requested():
                screen = self.get_target()
                if self._stop_requested():
                    break

                anchors = self.get_anchor_positions(screen)
                self._log("CAMERA", "visible_anchors", anchors=anchors)

                empty_fields = self.m.match_template(
                    field_img,
                    screen,
                    FIELD_MATCHING_THRESHOLD,
                )
                self.track_matches(empty_fields, screen)
                if empty_fields:
                    self._log("PLANT", "empty_fields", matches=len(empty_fields))
                    screen_changed = self.plant_crops(screen)
                    if screen_changed:
                        self.planted_crops += len(empty_fields)
                        self._log("PLANT", "screen_changed_rescan_required")
                        if self._wait(0.5):
                            break
                        continue
                if self._stop_requested():
                    break

                screen = self.get_target()
                if self.check_silo_is_full(screen):
                    self._log("SILO", "full")
                    self.silo_is_full = True
                self.restock_roadside_stand(screen)
                if self._stop_requested():
                    break

                if self.harvested_plants or self.silo_is_full:
                    self.silo_is_full = False
                    self.sell_items(screen)
                self.harvested_plants = False
                if self._stop_requested():
                    break

                grown_result = self._match_details(
                    plant_img,
                    screen,
                    WHEAT_MATCHING_THRESHOLD,
                )
                self.track_matches(grown_result.matches, screen)
                if not self.silo_is_full:
                    if grown_result.matches:
                        self._log(
                            "HARVEST",
                            "grown_plants",
                            matches=len(grown_result.matches),
                            raw_matches=grown_result.raw_match_count,
                            confidence=f"{grown_result.best_confidence:.3f}",
                            template_scale=f"{grown_result.template_scale:.2f}",
                        )
                        self.harvest_plants(screen)
                    else:
                        self._report_no_wheat(screen, grown_result)
                        if self._wait(10.0):
                            break

                if self._stop_requested():
                    break

                if self._wait(3.0):
                    break
            self._log("SESSION", "stopped")
            return True
        except Exception as error:
            self._log("SESSION", "worker_error", error=repr(error))
            raise
        finally:
            self._loop_lock.release()
