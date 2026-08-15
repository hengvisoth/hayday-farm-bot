# Hay Day Farm Bot

A Python desktop bot that automates a simple wheat farming loop in Hay Day.

The bot captures the screen, finds game objects with OpenCV, and controls the mouse to plant, harvest, and sell wheat. A CustomTkinter window shows bot logs and the latest marked screenshot.

> [!WARNING]
> This program controls your real mouse and keyboard. Test it carefully. Keep sensitive windows closed and be ready to press `q` to stop the bot.

## Features

- Finds empty fields and plants wheat.
- Finds mature wheat and harvests it.
- Detects a full silo.
- Collects sold items and creates new market offers.
- Uses image templates instead of fixed positions for most actions.
- Shows detection results and status messages in a desktop window.

## Requirements

- Python 3
- A visible Hay Day game window
- A 1920 x 1080 screen layout
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

The Stop button does not currently stop the active worker thread. Use the `q` key.

Your operating system may ask for screen recording, accessibility, input monitoring, or administrator permission.

## How It Works

The bot repeats this process:

1. Capture the full screen with `mss`.
2. Find objects with OpenCV template matching.
3. Plant wheat when empty fields are found.
4. Sell wheat after a harvest or when the silo is full.
5. Harvest mature wheat.
6. Close unexpected open windows.
7. Wait and repeat.

`pyautogui` performs mouse actions. The `keyboard` package checks for the `q` stop key.

## Project Structure

```text
.
├── main.py               # Application entry point
├── app.py                # Desktop window and log output
├── bot.py                # Farming loop and game actions
├── matcher.py            # OpenCV matching and drag paths
├── requirements.txt      # Python dependencies
├── templates/            # Images used for detection
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

The project does not have an automated test suite.

The scripts and screenshots in `test-environments/` are manual OpenCV checks. They help inspect matches and tune thresholds. Some scripts use fragile relative paths, so you may need to adjust the working directory or Python import path.

## Current Limits

- Only the wheat farming flow is supported.
- Screen capture is fixed to 1920 x 1080.
- Some camera anchors and coordinates are fixed.
- Template matching can produce false matches.
- The UI Stop button does not stop the worker thread.
- Recovery from unexpected game states is limited.
- Dependencies are not pinned to exact versions.

## Safety and Game Rules

This project can click the wrong place when the screen does not match its templates. Do not use the computer for other work while the bot is active.

Game automation may violate the game's terms of service. Check the current rules before using this project. You are responsible for how you use it.
