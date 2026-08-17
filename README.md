# Hay Day Farm Bot

A Python desktop bot that automates a simple wheat farming loop in Hay Day.

The bot captures the screen, finds game objects with OpenCV, and controls the mouse to plant, harvest, and sell wheat. A CustomTkinter window shows bot logs and the latest marked screenshot.

> [!WARNING]
> This program controls your real mouse and keyboard. Test it carefully. Keep sensitive windows closed and be ready to press `q` to stop the bot.

## Features

- Finds empty fields and plants wheat.
- Finds mature wheat and harvests it.
- Selects the scythe nearest to the clicked wheat.
- Searches several template scales for different game zoom and display sizes.
- Works without boat or market anchors when the game view stays stable.
- Drags through detected wheat centers instead of a large fixed boundary.
- Plants by clicking the field to reveal the wheat tool, then holding it and dragging through every detected field tile in one motion.
- Detects a full silo.
- Collects sold items and creates new market offers.
- Collects money from the roadside stand and restocks it with wheat once the OCR'd quantity exceeds a threshold.
- Uses image templates instead of fixed positions for most actions.
- Shows detection results and status messages in a desktop window.
- Writes detailed session logs and failure screenshots.

## Requirements

- Python 3
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed (used to read the wheat quantity badge before restocking the roadside stand). On Windows: `winget install --id UB-Mannheim.TesseractOCR -e`. `bot.py` looks for it at `C:\Program Files\Tesseract-OCR\tesseract.exe` by default; if it's installed elsewhere, set `pytesseract.pytesseract.tesseract_cmd` yourself or add the install directory to `PATH`.
- A visible Hay Day game window
- Permission to capture the screen and control the mouse and keyboard

The current templates and coordinates depend on the game scale, graphics, camera position, and screen resolution used when they were created.

## Setup

Open a terminal in the project folder.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Using `python3 -m pip` also works when the `pip` command is not available in your shell.

## Run the Bot

Run this command from the project root:

```bash
python3 main.py
```

Then:

1. Place the game in the expected position.
2. Click **Start** in the bot window.
3. Watch the preview and log output.
4. Press `q` to stop the bot loop.

The **Stop** button also stops the active worker. A drag always releases the mouse button when it stops or fails.

Your operating system may ask for screen recording, accessibility, input monitoring, or administrator permission.

## How It Works

The bot repeats this process:

1. Capture the full screen with `mss`.
2. Find objects with OpenCV template matching.
3. Plant wheat when empty fields are found.
4. Sell wheat after a harvest or when the silo is full.
5. Click a wheat crop near the center of the detected crop group.
6. Poll for the scythe for up to four seconds.
7. Re-detect wheat fresh on the post-click screen (never reuse pre-click positions, so a shifted camera can't produce a stale plan).
8. Reject a scythe/tool match that's implausibly far from the click instead of dragging to it.
9. Look for the game's own "selected" marker (`field_selected.png` / `wheat_selected.png`) to anchor the sweep on the exact tile that was clicked, falling back to the detected group's center if the marker isn't visible.
10. Hold the nearest scythe and drag row by row through every detected wheat tile, starting from that anchor.
11. Wait and repeat.

The bot does not click a generic close button after every loop. Recovery clicks only run for a known failure.

`pyautogui` performs mouse actions. The `keyboard` package checks for the `q` stop key.

## Project Structure

```text
.
├── main.py               # Application entry point
├── app.py                # Desktop window and log output
├── bot.py                # Farming loop and game actions
├── planner.py            # Pure camera and harvest route logic
├── matcher.py            # OpenCV matching and drag paths
├── diagnostics.py        # Session logs and failure screenshots
├── requirements.txt      # Python dependencies
├── templates/            # Images used for detection
├── tests/                # Automated tests with fake input devices
└── test-environments/    # Saved screenshots and manual checks
```

See `AGENTS.md` for a detailed developer guide.

## Template Matching

Templates are grouped by purpose:

- `templates/environment/`: Fields and farm buildings
- `templates/plants/`: Mature and growing crops
- `templates/interface/`: Tools, buttons, market slots, and status icons

Matching thresholds are defined near the top of `bot.py`. If detection fails, check the template image, screen scale, and threshold. Test template changes with saved screenshots before using live mouse automation.

## Testing

Run the automated tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The bot tests use fake screen capture and mouse objects. They do not move the real mouse.

The scripts and screenshots in `test-environments/` are manual OpenCV checks. They help inspect matches and tune thresholds. Some scripts use fragile relative paths, so you may need to adjust the working directory or Python import path.

## Diagnostics

Each bot session writes a timestamped log under `diagnostics/`. Failed actions can also save marked screenshots there. Screenshots show detected matches, selected points, and the planned route. The bot keeps the latest 20 failure screenshots.

Detection logs include `template_scale`. This shows which resized template matched the current game view.

When reporting a new problem, send the latest session log and failure screenshot.

## Current Limits

- Only the wheat farming flow is supported.
- Screen capture size is detected from the primary display at startup, but camera anchors and templates are still tuned for a 1920 x 1080 layout.
- Templates still depend on the game scale and display setup.
- Template matching can produce false matches.
- Recovery from unexpected game states is limited.
- Dependencies are not pinned to exact versions.

## Safety and Game Rules

This project can click the wrong place when the screen does not match its templates. Do not use the computer for other work while the bot is active.

Game automation may violate the game's terms of service. Check the current rules before using this project. You are responsible for how you use it.
