# Adaptive Harvesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harvest detected wheat even when boat and market anchors are not visible, while adding safe stopping and useful diagnostics.

**Architecture:** Pure coordinate planning lives in `planner.py`. OpenCV detection returns matches and confidence. `Bot` owns the action state machine and injected screen/mouse dependencies. Diagnostics and the UI handle logging without making worker-thread Tk calls.

**Tech Stack:** Python 3, `unittest`, OpenCV, NumPy, MSS, PyAutoGUI, CustomTkinter

**Spec:** `docs/superpowers/specs/2026-08-16-adaptive-harvesting-design.md`

## Global Constraints

- Poll for the scythe every 0.2 seconds for no more than 4 seconds.
- Crop pairs must be mutual nearest neighbors within 200 pixels.
- Reliable crop shifts must be within 15 pixels of the median on both axes.
- Reject shifts larger than 25 percent of screen width or height.
- Use `(0, 0)` when no reliable movement evidence exists.
- Keep drag route segments at or below 25 pixels.
- Keep only the latest 20 diagnostic screenshots.
- Tests must never control the real mouse.

---

### Task 1: Pure harvest planner

**Files:**
- Create: `planner.py`
- Create: `tests/__init__.py`
- Create: `tests/test_planner.py`

**Interfaces:**
- Consumes: Match rectangles shaped as `(center_x, center_y, width, height)`.
- Produces: `CameraTranslation`, `select_center_match()`, `select_nearest_match()`, `estimate_camera_translation()`, `translate_points()`, and `build_drag_route()`.

- [ ] **Step 1: Write failing selection and translation tests**

```python
class PlannerTests(unittest.TestCase):
    def test_selects_crop_nearest_group_center(self):
        matches = [(0, 0, 10, 10), (40, 40, 10, 10), (100, 100, 10, 10)]
        self.assertEqual(select_center_match(matches), matches[1])

    def test_uses_shared_anchor_translation(self):
        result = estimate_camera_translation(
            {"boat": (100, 200)}, {"boat": (112, 193)}, [], [], (1920, 1080)
        )
        self.assertEqual((result.dx, result.dy, result.method), (12, -7, "shared_anchor"))

    def test_falls_back_to_stable_scene_without_anchors(self):
        result = estimate_camera_translation({}, {}, [(10, 10)], [], (1920, 1080))
        self.assertEqual((result.dx, result.dy, result.method), (0, 0, "stable_fallback"))
```

- [ ] **Step 2: Run the planner tests and verify missing imports fail**

Run: `.venv/bin/python -m unittest tests.test_planner -v`

Expected: FAIL because `planner` does not exist.

- [ ] **Step 3: Implement selection and camera translation**

```python
@dataclass(frozen=True)
class CameraTranslation:
    dx: int
    dy: int
    method: str
    confidence: str

def select_center_match(matches):
    if not matches:
        return None
    mean_x = sum(match[0] for match in matches) / len(matches)
    mean_y = sum(match[1] for match in matches) / len(matches)
    return min(matches, key=lambda match: (match[0] - mean_x) ** 2 + (match[1] - mean_y) ** 2)

def select_nearest_match(matches, point):
    if not matches:
        return None
    return min(matches, key=lambda match: (match[0] - point[0]) ** 2 + (match[1] - point[1]) ** 2)

def estimate_camera_translation(before_anchors, after_anchors,
                                before_crops, after_crops,
                                screen_size):
    for name in ("boat", "market"):
        if name in before_anchors and name in after_anchors:
            dx = after_anchors[name][0] - before_anchors[name][0]
            dy = after_anchors[name][1] - before_anchors[name][1]
            return validate_translation(dx, dy, "shared_anchor", screen_size)
    pairs = mutual_nearest_pairs(before_crops, after_crops, 200)
    if pairs:
        deltas = [(after[0] - before[0], after[1] - before[1]) for before, after in pairs]
        median_dx = round(median(delta[0] for delta in deltas))
        median_dy = round(median(delta[1] for delta in deltas))
        reliable = [delta for delta in deltas
                    if abs(delta[0] - median_dx) <= 15 and abs(delta[1] - median_dy) <= 15]
        if reliable:
            return validate_translation(median_dx, median_dy, "crop_positions", screen_size)
    return CameraTranslation(0, 0, "stable_fallback", "no_reliable_movement_evidence")

def translate_points(points, translation):
    return [(x + translation.dx, y + translation.dy) for x, y in points]
```

The implementation uses direct anchor deltas first. It then uses mutual nearest crop pairs. It validates the shift against the screen size. It returns the stable fallback last.

- [ ] **Step 4: Add failing route tests**

```python
def test_one_crop_route_reaches_exact_center(self):
    self.assertEqual(build_drag_route([(50, 0)], (0, 0), 25), [(25, 0), (50, 0)])

def test_route_visits_nearest_crop_first(self):
    route = build_drag_route([(100, 0), (20, 0), (60, 0)], (0, 0), 100)
    self.assertEqual(route, [(20, 0), (60, 0), (100, 0)])
```

- [ ] **Step 5: Run the route tests and verify `build_drag_route` is missing**

Run: `.venv/bin/python -m unittest tests.test_planner -v`

Expected: FAIL because `build_drag_route` is not defined.

- [ ] **Step 6: Implement nearest-neighbor routing and interpolation**

```python
def build_drag_route(crop_centers, start, max_segment=25):
    """Return points after start, including every exact crop center."""
```

Remove duplicate centers. Pick the closest remaining crop. Interpolate with `ceil(distance / max_segment)` equal steps.

- [ ] **Step 7: Run planner tests**

Run: `.venv/bin/python -m unittest tests.test_planner -v`

Expected: all planner tests PASS.

### Task 2: Detection confidence and diagnostic files

**Files:**
- Modify: `matcher.py`
- Create: `diagnostics.py`
- Create: `tests/test_matcher.py`
- Create: `tests/test_diagnostics.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `TemplateMatchResult(matches, best_confidence)` from `Matcher.match_template_details()`.
- Produces: `Diagnostics.log()` and `Diagnostics.save_failure()`.
- Consumes: A `sink(line)` callback for UI output.

- [ ] **Step 1: Write a failing confidence test using a real image match**

```python
def test_match_details_include_best_confidence(self):
    target = numpy.zeros((30, 30), dtype=numpy.uint8)
    target[10:15, 12:17] = numpy.array([[0, 20, 40, 20, 0]] * 5, dtype=numpy.uint8)
    template = target[10:15, 12:17].copy()
    result = Matcher().match_template_details(template, target, 0.99, grouping=False)
    self.assertGreaterEqual(result.best_confidence, 0.99)
    self.assertIn([14, 12, 5, 5], result.matches)
```

- [ ] **Step 2: Run the matcher test and verify the method is missing**

Run: `.venv/bin/python -m unittest tests.test_matcher -v`

Expected: FAIL with missing `match_template_details`.

- [ ] **Step 3: Add `TemplateMatchResult` and preserve `match_template()` compatibility**

```python
@dataclass(frozen=True)
class TemplateMatchResult:
    matches: list
    best_confidence: float

def match_template_details(self, template, target, matching_threshold=0.45,
                           grouping=True):
    scores = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    _, best_confidence, _, _ = cv2.minMaxLoc(scores)
    height, width = template.shape[:2]
    rows, columns = np.where(scores >= matching_threshold)
    matches = [[int(x + width / 2), int(y + height / 2), width, height]
               for x, y in zip(columns, rows)]
    if grouping:
        matches, _ = cv2.groupRectangles(matches, self.group_threshold, self.eps)
        matches = [list(map(int, match)) for match in matches]
    return TemplateMatchResult(matches, float(best_confidence))
```

`match_template()` returns only `.matches`, so planting and selling keep their current API.

- [ ] **Step 4: Write failing diagnostic log and screenshot-retention tests**

```python
def test_log_writes_session_file_and_sink(self):
    with tempfile.TemporaryDirectory() as folder:
        emitted = []
        diagnostics = Diagnostics(emitted.append, folder, clock=fixed_clock)
        line = diagnostics.log("HARVEST", "wheat_found", matches=3)
        self.assertIn("[HARVEST] wheat_found matches=3", line)
        self.assertEqual(emitted, [line])
        self.assertIn(line, Path(diagnostics.session_path).read_text(encoding="utf-8"))

def test_save_failure_keeps_only_latest_limit(self):
    with tempfile.TemporaryDirectory() as folder:
        diagnostics = Diagnostics(directory=folder, screenshot_limit=2, clock=fixed_clock)
        image = numpy.zeros((20, 20, 3), dtype=numpy.uint8)
        diagnostics.save_failure("first", image)
        diagnostics.save_failure("second", image)
        diagnostics.save_failure("third", image)
        self.assertEqual(len(list(Path(folder).glob("*.png"))), 2)
```

Use `tempfile.TemporaryDirectory()`, a fixed clock, and small NumPy images. Assert file effects and emitted lines.

- [ ] **Step 5: Run the diagnostic tests and verify the module is missing**

Run: `.venv/bin/python -m unittest tests.test_diagnostics -v`

Expected: FAIL because `diagnostics` does not exist.

- [ ] **Step 6: Implement structured logs and annotated screenshots**

```python
class Diagnostics:
    def __init__(self, sink=None, directory="diagnostics", screenshot_limit=20,
                 clock=None):
        self.sink = sink
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.screenshot_limit = screenshot_limit
        self.clock = clock or datetime.now
        self.session_path = self.directory / f"session-{self.clock():%Y%m%d-%H%M%S}.log"

    def log(self, state, message="", **fields):
        timestamp = self.clock().strftime("%H:%M:%S.%f")[:-3]
        values = " ".join(f"{name}={value}" for name, value in fields.items())
        line = f"[{timestamp}] [{state}] {message} {values}".rstrip()
        with self.session_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if self.sink:
            self.sink(line)
        return line

    def save_failure(self, reason, image, matches=(), selected=None, path=()):
        marked = image.copy()
        mark_diagnostic_image(marked, matches, selected, path)
        filename = self.directory / self.next_screenshot_name(reason)
        cv2.imwrite(str(filename), marked)
        self.remove_old_screenshots()
        return filename
```

Use millisecond timestamps. Write session logs as UTF-8. Mark matches, selected points, and paths on a copy. Delete the oldest PNG files after saving.

- [ ] **Step 7: Ignore diagnostic output and run both test files**

Run: `.venv/bin/python -m unittest tests.test_matcher tests.test_diagnostics -v`

Expected: all matcher and diagnostic tests PASS.

### Task 3: Adaptive and stoppable harvest state machine

**Files:**
- Modify: `bot.py`
- Create: `tests/test_bot.py`

**Interfaces:**
- Consumes: Planner functions and `Matcher.match_template_details()`.
- Consumes: Injected `capture()`, mouse object, monotonic clock, and wait function.
- Produces: `Bot.request_stop()`, `Bot.reset_stop()`, `Bot.harvest_plants()` returning a success boolean.

- [ ] **Step 1: Write failing bot tests with fake capture and mouse boundaries**

```python
def test_harvest_reaches_scythe_without_camera_anchors(self):
    bot, mouse = make_harvest_bot([SCYTHE_SCREEN])
    self.assertTrue(bot.harvest_plants(WHEAT_SCREEN))
    self.assertEqual(mouse.calls[0], ("click", 500, 400))
    self.assertIn(("moveTo", 560, 350), mouse.calls)
    self.assertIn(("mouseDown", "left"), mouse.calls)
    self.assertIn(("mouseUp", "left"), mouse.calls)

def test_harvest_waits_until_scythe_appears(self):
    bot, mouse = make_harvest_bot([NO_SCYTHE_SCREEN, SCYTHE_SCREEN])
    self.assertTrue(bot.harvest_plants(WHEAT_SCREEN))
    self.assertEqual(mouse.calls.count(("click", 500, 400)), 1)

def test_scythe_timeout_does_not_guess_a_click(self):
    bot, mouse = make_harvest_bot([NO_SCYTHE_SCREEN], timeout=0.4)
    self.assertFalse(bot.harvest_plants(WHEAT_SCREEN))
    self.assertEqual([call for call in mouse.calls if call[0] == "click"], [("click", 500, 400)])

def test_stop_during_wait_prevents_more_mouse_input(self):
    bot, mouse = make_harvest_bot([NO_SCYTHE_SCREEN], stop_on_wait=True)
    self.assertFalse(bot.harvest_plants(WHEAT_SCREEN))
    self.assertEqual(mouse.calls, [("click", 500, 400)])

def test_mouse_is_released_when_drag_raises(self):
    bot, mouse = make_harvest_bot([SCYTHE_SCREEN], raise_on_drag=True)
    with self.assertRaises(RuntimeError):
        bot.harvest_plants(WHEAT_SCREEN)
    self.assertEqual(mouse.calls[-1], ("mouseUp", "left"))
```

The fake matcher returns complete `TemplateMatchResult` values for named fake screens. The fake mouse records calls but never imports or calls the real controller. Assertions check Bot results and state, not fake existence.

- [ ] **Step 2: Run bot tests and verify the adaptive API fails**

Run: `.venv/bin/python -m unittest tests.test_bot -v`

Expected: FAIL because dependency injection, stop methods, and adaptive harvesting do not exist.

- [ ] **Step 3: Add injected boundaries and stop control**

```python
def __init__(self, logger, set_tracking_img, capture=None, mouse=None,
             key_state=None, diagnostics=None, matcher=None, clock=None,
             waiter=None):
    self.capture = capture or self._capture_screen
    self.mouse = mouse or pa
    self.key_state = key_state or keyboard.is_pressed
    self.diagnostics = diagnostics or Diagnostics(logger.log)
    self.m = matcher or Matcher()
    self.clock = clock or monotonic
    self.waiter = waiter or self.stop_event.wait

def request_stop(self): self.stop_event.set()
def reset_stop(self): self.stop_event.clear()
```

All long waits use the stop-aware waiter. The loop checks the event before actions.

- [ ] **Step 4: Implement adaptive harvesting**

The exact state order is `DETECT_WHEAT`, `OPEN_TOOL`, `WAIT_FOR_SCYTHE`, `PLAN_DRAG`, and `DRAG`. Use the initial screen for wheat and anchors. Poll fresh screens for the scythe. Estimate translation from shared anchors or crops. Use stable `(0, 0)` when those are absent. Select the nearest scythe and drag through the planned crop centers.

- [ ] **Step 5: Make drag release unconditional after mouse-down**

```python
self.mouse.mouseDown(button="left")
try:
    for point in route:
        if self._stop_requested():
            return False
        self.mouse.moveTo(point[0], point[1], duration=0.1, _pause=False)
    return True
finally:
    self.mouse.mouseUp(button="left")
```

- [ ] **Step 6: Remove generic close recovery from every loop iteration**

Keep recovery callable only from a named failure branch. Log its reason before input.

- [ ] **Step 7: Run bot tests**

Run: `.venv/bin/python -m unittest tests.test_bot -v`

Expected: all bot tests PASS without real mouse input.

### Task 4: Thread-safe UI and duplicate-worker guard

**Files:**
- Modify: `app.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Produces: `WorkerController.start()` returning `True` only when a worker starts.
- Produces: `WorkerController.stop()` signaling `Bot.request_stop()`.
- Consumes: Logger and preview queues drained by `App` through `after()`.

- [ ] **Step 1: Write failing worker-controller tests**

```python
class FakeThread:
    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False
    def start(self):
        self.started = True
    def is_alive(self):
        return self.started

def test_start_refuses_second_live_worker(self):
    controller = WorkerController(FakeBot(), thread_factory=FakeThread)
    self.assertTrue(controller.start())
    self.assertFalse(controller.start())

def test_stop_signals_bot(self):
    bot = FakeBot()
    controller = WorkerController(bot, thread_factory=FakeThread)
    controller.start()
    controller.stop()
    self.assertTrue(bot.stop_requested)
```

Use a fake thread factory whose thread reports `is_alive()`. Assert the controller return value and bot state changes.

- [ ] **Step 2: Run the worker tests and verify the controller is missing**

Run: `.venv/bin/python -m unittest tests.test_worker -v`

Expected: FAIL because `WorkerController` does not exist.

- [ ] **Step 3: Implement `WorkerController` and queue-based UI updates**

```python
class WorkerController:
    def start(self):
        if self.is_running():
            return False
        self.bot.reset_stop()
        self.thread = self.thread_factory(target=self.bot.bot_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.bot.request_stop()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()
```

`Logger.log()` places strings in a queue. `App._drain_ui_queues()` inserts log lines at `end`, scrolls to `end`, applies the newest preview image, updates button state, and schedules itself again.

- [ ] **Step 4: Run worker and full tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: adaptive anchor fallback, Stop behavior, diagnostics location, and automated test command.

- [ ] **Step 1: Update README behavior and test instructions**

Replace the old Stop-button limitation. Explain that the bot uses anchors, crop motion, then stable zero movement. Add `python -m unittest discover -s tests -v` and explain `diagnostics/` retention.

- [ ] **Step 2: Run syntax checks**

Run: `.venv/bin/python -m compileall -q .`

Expected: exit code 0.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS with no mouse movement.

- [ ] **Step 4: Check the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only planned source, test, documentation, and ignore files are changed.
