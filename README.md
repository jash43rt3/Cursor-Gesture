# Hand Gesture Virtual Mouse

A webcam-based virtual mouse that lets you control your computer cursor using hand gestures. Built with Python, OpenCV, and MediaPipe — no extra hardware needed beyond a standard webcam.

## What it does

The application tracks your hand in real time through the webcam feed and maps specific gestures to mouse actions. It uses MediaPipe's 3D world landmarks so detection stays consistent regardless of hand rotation or distance from the camera.

### Supported gestures

| Gesture | Action |
|---|---|
| Index finger up, others folded | Move cursor |
| Thumb + Index pinch (quick) | Left click |
| Thumb + Index pinch (hold ~0.4s) | Drag & drop |
| Thumb + Middle pinch | Right click |
| Thumb + Ring pinch | Double click |
| Index + Middle up (peace sign), move vertically | Scroll |
| Thumb + Pinky pinch | Screenshot (saved to `./Screenshots`) |
| Closed fist / open palm | Rest (no action) |

### Keyboard controls (camera window)

- **P** — Pause / resume tracking
- **D** — Toggle debug overlay (shows finger angles and pinch distances)
- **Q** or **ESC** — Quit

## Setup

### Prerequisites

- Python 3.8+
- A working webcam
- Windows, macOS, or Linux

### Installation

```bash
pip install -r requirements.txt
```

The pinned dependencies are:

```
opencv-python
mediapipe==0.10.14
numpy<2
pyautogui
```

> **Note:** `numpy<2` is required because MediaPipe 0.10.14 is not yet compatible with NumPy 2.x.

### Running

```bash
python hand_mouse.py
```

If your webcam isn't at index 0, pass the index as an argument:

```bash
python hand_mouse.py 1
```

On Windows, the app tries the DirectShow backend first (`CAP_DSHOW`) for faster camera initialization, then falls back to the default backend automatically.

## How it works

### Pointer tracking

The cursor position is derived from the **index knuckle** (landmark 5) rather than the fingertip. This keeps the pointer stable during pinch gestures — if the fingertip were used, the act of pinching would shake the cursor.

Raw landmark positions are smoothed with a **1-Euro filter**, which adapts its cutoff frequency based on movement speed. The result is low jitter when the hand is still and minimal lag when it's moving quickly. An offset-decay mechanism gradually re-aligns the cursor with the hand after periods of lock (e.g., during clicks), preventing sudden jumps.

### Gesture classification

Finger extension is determined by computing joint angles across the MCP, PIP, DIP, and TIP landmarks. Hysteresis thresholds (150° to extend, 135° to fold) prevent flickering when a finger is near the boundary.

Pinch detection uses the ratio of thumb-tip-to-finger-tip distance relative to palm size, making it scale-invariant. A separate "approaching pinch" threshold locks the cursor early so it doesn't drift while the user is about to click. Fingers that are tightly folded below the `FOLD_ANGLE` are excluded from pinch eligibility, which prevents false triggers from fists or resting hand positions.

All gesture transitions require a debounce period (default: 2 consecutive frames) before being accepted.

### Action handling

- **Left click vs. drag**: A quick pinch-and-release fires a left click. Holding the pinch beyond `DRAG_HOLD_TIME` (0.4s) initiates a drag; releasing completes the drop.
- **Cooldown**: Right click, double click, and screenshot share a 0.6s cooldown to prevent accidental repeats.
- **Post-click freeze**: After any pinch ends, the cursor locks briefly (0.2s) so the release motion doesn't cause unintended movement.
- **Scroll**: Activated by the peace-sign gesture. Vertical hand displacement from an anchor point drives scroll amount, with a deadzone around center and capped maximum speed.
- **Hand lost**: If no hand is detected for 8 consecutive frames, all active states (drag, scroll anchor, filters) are fully reset.

## Tuning

All configurable parameters live at the top of `hand_mouse.py`. Press **D** during runtime to see live finger angles and pinch distances, which makes tuning much easier. Key values to adjust:

- `FRAME_MARGIN` — Size of the inactive border inside the camera frame. Larger = smaller active area.
- `ONE_EURO_MIN_CUTOFF` / `ONE_EURO_BETA` — Filter responsiveness. Lower min_cutoff = steadier at rest; higher beta = less lag during fast movement.
- `PINCH_ENTER` / `PINCH_EXIT` — Pinch sensitivity. Narrow the gap for snappier response, widen it for more stability.
- `DRAG_HOLD_TIME` — How long to hold before drag activates. Increase if drags trigger too easily.
- `ACTION_COOLDOWN` — Minimum interval between discrete actions.

## Project structure

```
IvpProject/
├── hand_mouse.py       # Main application (all logic in one file)
├── requirements.txt    # Python dependencies
├── run.exe             # Pre-built Windows executable
└── Screenshots/        # Captured screenshots (auto-created at runtime)
```

## Known limitations

- Requires adequate lighting and a clear view of one hand. Performance degrades in low light or with heavy background clutter.
- Single-hand only — two-hand gestures are not supported.
- PyAutoGUI cannot control the cursor over certain elevated/admin windows on some operating systems.
- Frame rate depends on CPU performance; expect 20–30 FPS on typical laptops.
