# Hay Day Farm Bot: Project Guide

## Project Purpose

This project automates a simple wheat farming loop in Hay Day.

The bot watches a fixed part of the screen. It uses image templates to find game objects and interface controls. It then moves and clicks the mouse to plant wheat, harvest wheat, and sell wheat.

The project also has a desktop window. The window starts the bot, shows its latest marked screenshot, and prints log messages.

## Main Workflow

The bot repeats this loop:

1. Capture the full 1920 x 1080 screen.
2. Find empty fields and plant wheat.
3. Sell wheat after a harvest or when the silo is full.
4. Find mature wheat and harvest it.
5. Check for a full silo or an unexpected open window.
6. Wait, then repeat.

Press `q` to stop the bot loop.

## Project Structure

- `main.py`: Starts the desktop application.
- `app.py`: Defines the CustomTkinter window, log panel, screen preview, and Start/Stop buttons.
- `bot.py`: Contains the farming loop and all game actions.
- `matcher.py`: Finds image templates with OpenCV. It also creates field boundaries, drag paths, and preview markings.
- `math.py`: Contains an unused distance helper. `matcher.py` uses Python's standard `math` module instead.
- `templates/`: Small reference images used to detect fields, crops, buildings, and interface controls.
- `test-environments/`: Screenshots and manual OpenCV scripts used to tune and inspect template matching.
- `README.md`: Contains the original short setup notes.

## Runtime Flow

`main.py` creates `App`.

`App` creates a `Bot` and gives it two callbacks:

- A logger callback for status messages.
- An image callback for the marked screen preview.

The Start button runs `Bot.bot_loop()` in a background thread. The bot captures the screen with `mss`, detects objects with `Matcher`, and controls the game with `pyautogui` and `keyboard`.

## Image Matching

All detection uses `cv2.matchTemplate()` with `TM_CCOEFF_NORMED`.

Each object has a matching threshold near the top of `bot.py`. A lower threshold accepts weaker matches. A higher threshold needs a closer match.

Matches are stored as:

```text
[center_x, center_y, width, height]
```

`Matcher.match_template()` can group nearby results. The bot uses the results for clicks and drag paths.

The boat and market images are camera anchors, logged for diagnostics. Field and wheat routes are not adjusted through them — after a click, the bot re-detects the field/wheat tiles fresh on the new screen instead of translating pre-click positions, so a shifted camera never produces a stale plan.

## Templates

Template folders have these roles:

- `templates/environment/`: Farm buildings and empty fields.
- `templates/plants/`: Mature and growing crops.
- `templates/interface/`: Buttons, tools, market slots, and status icons.

Template images depend on the game's scale, screen resolution, graphics, and camera position. If detection fails, first check the template image and its threshold.

All templates load through `_load_template()` in `bot.py`, which converts them to 4-channel BGRA if needed. `mss` screen captures are always BGRA, and `cv2.matchTemplate` requires the template and target to share a channel count -- a template re-saved by an image editor without an alpha channel would otherwise crash `matchTemplate` (not just fail to match) the moment it's used. Always add new templates through `_load_template()`, not a raw `cv2.imread()`.

## Setup

Use Python 3. Install these packages:

```bash
python -m pip install numpy opencv-python customtkinter mss pillow keyboard pyautogui pytesseract
```

`pytesseract` is a wrapper only -- it also needs the Tesseract-OCR engine installed separately (`winget install --id UB-Mannheim.TesseractOCR -e` on Windows). `bot.py` points `pytesseract.pytesseract.tesseract_cmd` at `C:\Program Files\Tesseract-OCR\tesseract.exe` if that path exists; otherwise it relies on `tesseract` being on `PATH`.

The repository has no dependency lock file or package metadata.

Run commands from the repository root because template paths are relative:

```bash
python main.py
```

The original README says to run with administrator rights. This may be needed for global keyboard input or mouse control on some systems.

## Tests and Manual Checks

There is no automated test suite.

Files in `test-environments/` are manual visual checks. They load a saved screenshot, run one or more template matches, draw the results, and open an OpenCV window.

These scripts use fragile relative paths. Check their current working directory and Python import path before running them.

For changes to image matching:

1. Test against the saved screenshots.
2. Confirm that correct objects are marked.
3. Check for false matches.
4. Test with the live game before relying on mouse automation.

## Important Assumptions

- The captured screen starts at `(0, 0)`.
- The screen size is detected at startup from the primary display (`pyautogui.size()`), not hardcoded.
- Mouse coordinates use the same coordinate system as captured images.
- The game is visible and not covered by another window.
- The game scale and graphics look like the saved templates.
- Template files can be loaded from paths relative to the repository root.
- Wheat is the only crop handled by the main bot flow.

## Current Limits

- Screen capture size is detected, but `BOAT_ANCHOR`/`MARKET_ANCHOR` and the saved templates are still tuned for a 1920 x 1080 layout. Running at another resolution can still misalign camera-position estimation and matching.
- There is no configuration file or command-line interface.
- There are no automated tests.
- Image files are loaded when `bot.py` is imported. Missing files can cause later matching errors.
- The UI Stop button does not stop the active worker thread. Pressing `q` is the working stop signal.
- UI updates are called from the worker thread. Tkinter normally expects UI work on the main thread.
- The bot has no pause, retry limit, or recovery state for many unexpected game states.
- The bot may click the wrong place when a template produces a false match.

## Safety

This program controls the real mouse and keyboard.

Before running it:

- Keep the game in the expected position.
- Close or move sensitive windows.
- Be ready to press `q`.
- Test new templates on saved screenshots first.
- Do not run untested automation while doing other work on the computer.

Automation may violate a game's terms of service. The user is responsible for checking the rules and accepting the risk.

## Change Guidelines

- Keep screen capture, image matching, bot decisions, and UI code separate.
- Put reusable detection logic in `matcher.py`.
- Put game actions and state in `bot.py`.
- Put desktop interface work in `app.py`.
- Avoid adding more fixed coordinates when a detected anchor can be used.
- Keep each template threshold close to its template definition.
- Use clear names for templates, thresholds, and game states.
- Add automated tests for pure matching and path functions when changing them.
- Do not run live mouse automation as part of an automated test.
- Update this file when the architecture, setup, or main workflow changes.
