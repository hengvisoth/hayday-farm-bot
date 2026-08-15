# Adaptive Harvesting Design

## Goal

Make harvesting work when the boat and market camera anchors are not visible. Improve the harvest path, stop behavior, recovery logic, and diagnostic output.

The current bot detects wheat correctly. It clicks the wheat correctly. It also detects the scythe. It then stops before touching the scythe because `get_camera_pos()` returns `(0, 0)`. The user's camera does not move after the wheat click, so this anchor requirement is incorrect for that device.

## Scope

This change will:

- Make camera anchors optional.
- Estimate movement from anchors or crop positions.
- Use zero movement when the scene remains stable.
- Wait for the scythe instead of using one fixed delay.
- Select the scythe nearest to the clicked wheat.
- Drag through detected wheat centers.
- Add structured logs and failure screenshots.
- Make the Stop button stop the worker.
- Prevent two bot loops from running at the same time.
- Make mouse release safe during errors and stops.
- Limit automatic recovery clicks to confirmed failures.
- Add tests that never control the real mouse.

This change will not add new crops, selling strategies, multi-monitor selection, or automatic template generation.

## Components

### `planner.py`

This new module will contain pure coordinate logic. It will not capture the screen or control input.

It will provide:

- A function that selects the crop nearest to the center of the crop group.
- A function that selects the scythe match nearest to the clicked crop.
- A function that estimates camera translation.
- A function that translates old crop coordinates to the current screen.
- A function that orders crop centers into a short drag route.
- A function that adds intermediate points to long route segments.

Keeping these functions pure makes them easy to test with small coordinate lists.

### `matcher.py`

`Matcher` will remain responsible for image detection and preview marks.

It will add a small result helper that reports:

- Grouped matches.
- The best raw confidence score.

This allows the bot to log match quality without mixing logging into image detection.

The old boundary path functions will remain only while planting still uses them. Harvesting will use the new center-based route.

### `bot.py`

`Bot` will control an explicit action sequence. Harvesting will have named states such as `DETECT_WHEAT`, `OPEN_TOOL`, `WAIT_FOR_SCYTHE`, `PLAN_DRAG`, and `DRAG`.

Each state will use a captured screen passed into its helpers. Camera functions will not capture extra hidden screenshots. This keeps detection results from the same moment together.

`Bot` will own a `threading.Event` for stopping. Long waits will use the event, so Stop works without waiting for a full loop.

### `diagnostics.py`

This new module will format logs and manage diagnostic files.

It will write one session log under `diagnostics/`. It will also save an annotated screenshot when an action times out or produces an invalid plan. It will keep the latest 20 screenshots and remove older diagnostic screenshots.

The `diagnostics/` folder will be ignored by Git.

### `app.py`

The UI will show logs from oldest to newest. New messages will appear at the bottom.

The UI will use a queue to receive messages from the bot thread. The Tkinter thread will drain the queue. This avoids direct widget updates from the worker thread.

Start will refuse to create another worker while one is alive. Stop will signal the bot's stop event.

## Harvest Data Flow

1. Capture one screen.
2. Detect mature wheat and record match count and confidence.
3. Select the wheat match nearest to the group center.
4. Record available camera anchors from the same screen.
5. Click the selected wheat center.
6. Poll for the scythe every 0.2 seconds for up to 4 seconds.
7. When the scythe appears, keep the screen and all scythe candidates from that poll.
8. Detect mature wheat again on the scythe screen.
9. Estimate camera translation.
10. Translate the original wheat centers.
11. Select the scythe candidate nearest to the clicked wheat.
12. Build a route through translated wheat centers.
13. Press on the scythe, drag through the route, and release.
14. Log the result and return to the main loop.

Fixed sleeps will remain only for short input settling where polling cannot verify the result.

## Camera Translation

The planner will return `dx`, `dy`, a method name, and a confidence description.

The methods will be tried in this order:

1. **Shared anchor:** If the same boat or market anchor is visible before and after the wheat click, use the anchor position difference.
2. **Crop positions:** Pair mutual nearest crop centers that are no more than 200 pixels apart. Use the median `after - before` difference. A pair is reliable when its difference is within 15 pixels of the median on both axes.
3. **Stable fallback:** If no reliable anchor or crop pair exists, use `(0, 0)` and log a warning. This supports the confirmed device behavior where the camera stays still.

At least one reliable pair is required. A shift larger than 25 percent of the screen width or height will be rejected and replaced by the stable fallback. Every decision will be logged.

## Crop and Scythe Selection

The initial wheat click will use the crop nearest to the crop group's center. This avoids selecting an edge crop that may put the tool menu near a screen boundary.

The scythe selection will use the candidate nearest to the clicked wheat after translation. The bot will log every candidate center, the best confidence score, and the selected center.

If no scythe appears before the timeout, the bot will save a screenshot and stop that harvest attempt. It will not guess a coordinate.

## Drag Route

The harvest route will pass through detected wheat centers.

The route builder will:

1. Remove duplicate centers.
2. Start with the crop center nearest to the scythe.
3. Repeatedly select the nearest unvisited crop center.
4. Add intermediate points so no route segment is longer than 25 pixels.

A single detected wheat match will produce a route containing that exact center. The bot will no longer create a large boundary around one crop.

The mouse sequence will be:

1. Move to the scythe center.
2. Press the left mouse button.
3. Move to each route point.
4. Release the left mouse button.

The release will be in `finally`. It will run after success, an exception, or a stop request.

## Logging

Logs will use this shape:

```text
[14:22:10.125] [HARVEST] wheat_matches=6 confidence=0.93 selected=(923,512)
[14:22:10.780] [HARVEST] scythe_matches=1 confidence=0.91 selected=(1012,430)
[14:22:10.781] [CAMERA] method=crop_positions shift=(0,0) confidence=stable
[14:22:10.782] [DRAG] points=6 start=(1012,430) end=(1080,620)
```

The session start will log:

- Screen capture bounds.
- PyAutoGUI screen size.
- Template load status.
- Important thresholds and timeouts.

Each action will log:

- State name.
- Match counts.
- Best confidence score.
- Candidate and selected coordinates.
- Camera method and shift.
- Route point count, start, and end.
- Mouse down and mouse up.
- Stop requests.
- Timeout and recovery decisions.

Failure screenshots will contain match boxes, selected points, and the planned path when one exists.

## Recovery and Safety

The bot will not search for and click a generic close button at the end of every loop. The current low close threshold can close the harvest interface after an aborted action.

Recovery will run only after a known action timeout or invalid state. It will log the reason before clicking anything.

The stop event will be checked:

- Before every mouse action.
- During polling.
- Between drag points.
- During long waits.

If Stop is requested during a drag, the left mouse button will be released before the worker exits.

## Tests

Tests will use Python's built-in `unittest` package. No test dependency is needed.

Pure planner tests will cover:

- Center crop selection.
- Nearest scythe selection.
- Shared-anchor translation.
- Crop-position translation.
- Zero stable fallback.
- Rejection of invalid large shifts.
- One-crop drag routes.
- Multi-crop nearest-neighbor routes.
- Intermediate route points.

Bot behavior tests will use fake capture and input objects. They will cover:

- Waiting until a scythe appears.
- Scythe timeout handling.
- Stop during a wait.
- Stop during a drag.
- Mouse release after an exception.
- Refusal to start a second worker.

Tests will not import or call the real mouse controller for actions.

## Success Criteria

- The bot reaches the scythe when camera anchors are absent.
- A stable scene produces a `(0, 0)` camera shift.
- The drag route passes through detected wheat centers.
- Stop ends the active loop and does not leave the mouse pressed.
- Start cannot create two bot loops.
- Failure logs provide coordinates, confidence, state, and camera method.
- Failed actions save useful screenshots.
- All automated tests pass without moving the real mouse.
