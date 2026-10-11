"""
Hand Gesture Virtual Mouse  v2  (IVP Project)
=============================================
Control your computer mouse with your hand using a webcam.

Libraries : OpenCV (capture / drawing), MediaPipe Hands (21 landmarks),
            PyAutoGUI (mouse + screenshot), NumPy.

GESTURES (one hand, palm facing the camera)
-------------------------------------------
 MOVE CURSOR   : Index finger straight up, middle finger folded
                 (cursor follows your INDEX KNUCKLE, so pinching never shakes it)
 LEFT CLICK    : Quick pinch  -> Thumb + Index
 DRAG & DROP   : Pinch Thumb + Index and HOLD (~0.4s), move hand, open to drop
 RIGHT CLICK   : Pinch Thumb + Middle
 DOUBLE CLICK  : Pinch Thumb + Ring
 SCROLL        : Index + Middle up (peace sign), move hand UP / DOWN
 SCREENSHOT    : Pinch Thumb + Pinky  (saved in ./Screenshots)
 REST          : Closed fist or open palm (nothing happens)

KEYS (camera window)
--------------------
 P = pause / resume     D = debug numbers (angles + pinch distances)
 Q or ESC = quit
"""

import math
import os
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ----------------------------------------------------------------------------
# SETTINGS  (tweak these to taste - press D in the window to see live numbers)
# ----------------------------------------------------------------------------
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CAM_W, CAM_H = 640, 480
FRAME_MARGIN = 90          # active area margin (px) inside the camera frame

# --- pointer ---------------------------------------------------------------
POINTER_LM = 5             # 5 = index knuckle (stable). 8 = fingertip (old, shaky)
ONE_EURO_MIN_CUTOFF = 1.2  # lower = steadier when slow
ONE_EURO_BETA = 12.0       # higher = less lag when moving fast
MIN_MOVE_PX = 2            # ignore cursor moves smaller than this (kills jitter)
OFFSET_DECAY = 0.90        # how fast cursor re-joins the hand after a freeze
POST_CLICK_FREEZE = 0.20   # seconds the cursor stays locked after a pinch ends

# --- gesture detection -----------------------------------------------------
EXT_ON, EXT_OFF = 150, 135 # finger-straightness angle (deg): becomes UP above ON, DOWN below OFF
FOLD_ANGLE = 110           # a finger bent tighter than this cannot pinch
PINCH_ENTER = 0.35         # thumb-tip to finger-tip distance / palm size to start pinch
PINCH_EXIT = 0.55          # ... and to end it (hysteresis stops flicker)
PINCH_PRE = 0.60           # cursor locks as soon as a pinch is this close
LANDMARK_SMOOTH = 0.6      # 1.0 = raw landmarks, lower = smoother
DEBOUNCE_FRAMES = 2        # frames a gesture must hold before it is accepted
LOST_FRAMES = 8            # frames without a hand before everything is released

# --- actions ---------------------------------------------------------------
DRAG_HOLD_TIME = 0.40      # seconds of holding pinch before drag starts
ACTION_COOLDOWN = 0.60     # seconds between right click / double click / screenshot
SCROLL_DEADZONE = 0.04     # fraction of frame height ignored around the anchor
SCROLL_GAIN = 1200         # scroll speed (Windows wheel units per frame)
SCROLL_MAX = 400
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Screenshots")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Landmark ids
WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9
FINGERS = ("index", "middle", "ring", "pinky")
JOINTS = {  # (MCP, PIP, DIP, TIP)
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
PINCH_ACTION = {"index": "left", "middle": "right", "ring": "double", "pinky": "screenshot"}
PINCH_GESTURES = tuple(PINCH_ACTION.values())


# ----------------------------------------------------------------------------
# One Euro filter: smooth when slow (no jitter), responsive when fast (no lag)
# ----------------------------------------------------------------------------
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = None
        self.x_prev = 0.0
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


# ----------------------------------------------------------------------------
# Gesture detection (uses MediaPipe WORLD landmarks = real 3D, so it does not
# care about hand rotation or distance from the camera)
# ----------------------------------------------------------------------------
def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def finger_angle(W, f):
    """180 = perfectly straight finger, ~90 = tightly folded."""
    mcp, pip, dip, tip = JOINTS[f]
    return 0.5 * (joint_angle(W[mcp], W[pip], W[dip]) + joint_angle(W[pip], W[dip], W[tip]))


class GestureClassifier:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ext = {f: False for f in FINGERS}
        self.pinch = None
        self.debug = {}

    def update(self, W):
        """W = 21x3 array of 3D landmarks. Returns (gesture, approaching_pinch)."""
        palm = float(np.linalg.norm(W[WRIST] - W[MIDDLE_MCP])) + 1e-6
        angles = {f: finger_angle(W, f) for f in FINGERS}
        ratios = {f: float(np.linalg.norm(W[THUMB_TIP] - W[JOINTS[f][3]])) / palm
                  for f in FINGERS}

        # finger up / down with hysteresis
        for f in FINGERS:
            self.ext[f] = angles[f] > (EXT_OFF if self.ext[f] else EXT_ON)

        # only fingers that are not tightly folded may pinch (a fist or a folded
        # middle finger resting on the thumb is NOT a pinch)
        eligible = {f: ratios[f] for f in FINGERS if angles[f] > FOLD_ANGLE}
        keep = (self.pinch is not None and self.pinch in eligible
                and eligible[self.pinch] < PINCH_EXIT)
        if not keep:
            self.pinch = None
            if eligible:
                best = min(eligible, key=eligible.get)
                if eligible[best] < PINCH_ENTER:
                    self.pinch = best
        approaching = (self.pinch is not None) or (
            bool(eligible) and min(eligible.values()) < PINCH_PRE)

        e = self.ext
        if self.pinch:
            gesture = PINCH_ACTION[self.pinch]
        elif e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
            gesture = "scroll"
        elif e["index"] and not e["middle"]:
            gesture = "move"
        else:
            gesture = "rest"

        self.debug = {"angles": angles, "ratios": ratios, "ext": dict(e)}
        return gesture, approaching


# ----------------------------------------------------------------------------
# Mouse controller
# ----------------------------------------------------------------------------
class HandMouse:
    def __init__(self):
        self.cursor = tuple(float(v) for v in pyautogui.position())
        self.mapped = self.cursor            # where the hand points (screen px)
        self.offset = (0.0, 0.0)
        self.need_anchor = True
        self.locked = False
        self.fx = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
        self.fy = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
        self.gesture = "rest"                # accepted (debounced) gesture
        self.candidate = "rest"
        self.count = 0
        self.pinch_start = 0.0
        self.dragging = False
        self.freeze_until = 0.0
        self.last_action_time = 0.0
        self.scroll_anchor = None
        self.message = ""
        self.message_until = 0.0
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # -- small helpers -------------------------------------------------------
    def notify(self, text, seconds=1.0):
        self.message = text
        self.message_until = time.time() + seconds

    def cooldown_ok(self):
        return time.time() - self.last_action_time > ACTION_COOLDOWN

    def take_screenshot(self):
        name = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        path = os.path.join(SCREENSHOT_DIR, name)
        pyautogui.screenshot().save(path)
        self.notify("Screenshot saved: " + name, 2.0)
        print("[Screenshot]", path)

    # -- pointer -------------------------------------------------------------
    def track_pointer(self, px, py, w, h, t):
        """Filter the pointer every frame (even while the cursor is locked)."""
        m = FRAME_MARGIN
        nx = float(np.clip((px - m) / (w - 2.0 * m), 0.0, 1.0))
        ny = float(np.clip((py - m) / (h - 2.0 * m), 0.0, 1.0))
        nx, ny = self.fx(nx, t), self.fy(ny, t)
        self.mapped = (nx * (SCREEN_W - 1), ny * (SCREEN_H - 1))

    def move_cursor(self, allow):
        if not allow:
            self.need_anchor = True
            self.locked = True
            return
        self.locked = False
        mx, my = self.mapped
        if self.need_anchor:
            # continue from where the cursor is - no sudden jump after a lock
            self.cursor = tuple(float(v) for v in pyautogui.position())
            self.offset = (self.cursor[0] - mx, self.cursor[1] - my)
            self.need_anchor = False
        x = min(max(mx + self.offset[0], 0.0), SCREEN_W - 1.0)
        y = min(max(my + self.offset[1], 0.0), SCREEN_H - 1.0)
        self.offset = (self.offset[0] * OFFSET_DECAY, self.offset[1] * OFFSET_DECAY)
        if math.hypot(x - self.cursor[0], y - self.cursor[1]) >= MIN_MOVE_PX:
            pyautogui.moveTo(x, y)
            self.cursor = (x, y)

    # -- gesture state machine ----------------------------------------------
    def update_gesture(self, new):
        if new == self.gesture:
            self.candidate, self.count = new, 0
            return
        if new == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = new, 1
        if self.count >= DEBOUNCE_FRAMES:
            self.set_gesture(new)

    def set_gesture(self, new):
        if new == self.gesture:
            return
        self.on_gesture_end(self.gesture)
        self.gesture = new
        self.candidate, self.count = new, 0
        self.on_gesture_start(new)

    def on_gesture_start(self, g):
        now = time.time()
        if g == "left":
            self.pinch_start = now
            self.dragging = False
        elif g == "right" and self.cooldown_ok():
            pyautogui.rightClick()
            self.last_action_time = now
            self.notify("Right Click")
        elif g == "double" and self.cooldown_ok():
            pyautogui.doubleClick()
            self.last_action_time = now
            self.notify("Double Click")
        elif g == "screenshot" and self.cooldown_ok():
            self.take_screenshot()
            self.last_action_time = now
        elif g == "scroll":
            self.scroll_anchor = None

    def on_gesture_end(self, g):
        now = time.time()
        if g == "left":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
                self.notify("Dropped")
            elif now - self.pinch_start < DRAG_HOLD_TIME:
                pyautogui.click()
                self.notify("Left Click")
        if g in PINCH_GESTURES:
            self.freeze_until = now + POST_CLICK_FREEZE

    # -- per-frame processing -----------------------------------------------
    def process(self, lm, w, h, raw_gesture, approaching, t):
        self.track_pointer(lm[POINTER_LM].x * w, lm[POINTER_LM].y * h, w, h, t)
        g = self.gesture

        # start the drag once the pinch has been held long enough
        if g == "left" and not self.dragging and t - self.pinch_start >= DRAG_HOLD_TIME:
            pyautogui.mouseDown()
            self.dragging = True
            self.notify("Dragging...", 60.0)

        if self.dragging:
            allow = True
        elif g == "move":
            freeze_now = approaching or raw_gesture in PINCH_GESTURES
            allow = (not freeze_now) and t >= self.freeze_until
        else:
            allow = False
        self.move_cursor(allow)

        if g == "scroll":
            self.do_scroll(lm, h)

    def do_scroll(self, lm, h):
        y = (lm[5].y + lm[9].y) / 2.0 * h
        if self.scroll_anchor is None:
            self.scroll_anchor = y
        dy = (self.scroll_anchor - y) / h        # + = hand moved up
        if abs(dy) > SCROLL_DEADZONE:
            amount = min((abs(dy) - SCROLL_DEADZONE) * SCROLL_GAIN, SCROLL_MAX)
            if os.name != "nt":
                amount /= 120.0
            n = max(1, int(round(amount)))
            pyautogui.scroll(n if dy > 0 else -n)

    def hand_lost(self):
        if self.gesture == "left" and not self.dragging:
            self.gesture = "rest"                # no accidental click
        self.set_gesture("rest")
        self.release_all()
        self.need_anchor = True
        self.fx.reset()
        self.fy.reset()
        self.scroll_anchor = None

    def release_all(self):
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False


# ----------------------------------------------------------------------------
# UI drawing
# ----------------------------------------------------------------------------
LABELS = {
    "move": "MOVE", "left": "LEFT CLICK / DRAG", "right": "RIGHT CLICK",
    "double": "DOUBLE CLICK", "scroll": "SCROLL", "screenshot": "SCREENSHOT",
    "rest": "REST",
}


def draw_overlay(frame, hand, ctrl, cls, w, h):
    lm = hand.landmark

    def pt(i):
        return int(lm[i].x * w), int(lm[i].y * h)

    cv2.circle(frame, pt(THUMB_TIP), 8, (255, 200, 0), -1)
    if cls.pinch:
        tip = pt(JOINTS[cls.pinch][3])
        cv2.circle(frame, tip, 8, (0, 255, 255), -1)
        cv2.line(frame, pt(THUMB_TIP), tip, (0, 255, 255), 2)
    color = (0, 0, 255) if ctrl.locked else (0, 255, 0)
    cv2.circle(frame, pt(POINTER_LM), 11, color, 2)


def draw_debug(img, cls):
    d = cls.debug
    if not d:
        return
    y = 62
    for f in FINGERS:
        txt = "%-6s angle %3d %-2s  thumb-dist %.2f" % (
            f, d["angles"][f], "UP" if d["ext"][f] else "", d["ratios"][f])
        cv2.putText(img, txt, (8, y), FONT, 0.5, (255, 255, 0), 1)
        y += 20
    cv2.putText(img, "pinch starts < %.2f, ends > %.2f" % (PINCH_ENTER, PINCH_EXIT),
                (8, y), FONT, 0.5, (200, 200, 200), 1)


def draw_hud(img, ctrl, fps, paused, hand_found):
    h, w = img.shape[:2]
    cv2.rectangle(img, (FRAME_MARGIN, FRAME_MARGIN),
                  (w - FRAME_MARGIN, h - FRAME_MARGIN), (255, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (w, 34), (30, 30, 30), -1)
    if paused:
        status, color = "PAUSED", (0, 165, 255)
    elif not hand_found:
        status, color = "NO HAND", (0, 0, 255)
    else:
        status, color = LABELS[ctrl.gesture], (0, 255, 0)
        if ctrl.locked and ctrl.gesture == "move":
            status += " [LOCKED]"
    cv2.putText(img, "Mode: " + status, (8, 24), FONT, 0.7, color, 2)
    cv2.putText(img, "FPS %d" % fps, (w - 90, 24), FONT, 0.6, (255, 255, 255), 2)
    if time.time() < ctrl.message_until:
        cv2.putText(img, ctrl.message, (8, h - 15), FONT, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "P pause  D debug  Q quit", (w - 235, h - 15), FONT, 0.5,
                (200, 200, 200), 1)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index %d). "
              "Try: python hand_mouse.py 1" % CAM_INDEX)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    ctrl = HandMouse()
    cls = GestureClassifier()
    paused, debug = False, False
    prev_time, fps = time.time(), 0.0
    lost, smoothed = 0, None

    print(__doc__)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received, exiting.")
                break
            frame = cv2.flip(frame, 1)                  # mirror view
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            now = time.time()

            hand_found = bool(result.multi_hand_landmarks
                              and result.multi_hand_world_landmarks)
            if hand_found:
                lost = 0
                hand = result.multi_hand_landmarks[0]
                world = result.multi_hand_world_landmarks[0]
                W = np.array([[p.x, p.y, p.z] for p in world.landmark], dtype=np.float32)
                smoothed = W if smoothed is None else (
                    LANDMARK_SMOOTH * W + (1.0 - LANDMARK_SMOOTH) * smoothed)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if not paused:
                    raw, approaching = cls.update(smoothed)
                    ctrl.update_gesture(raw)
                    ctrl.process(hand.landmark, w, h, raw, approaching, now)
                    draw_overlay(frame, hand, ctrl, cls, w, h)
            else:
                lost += 1
                if lost == LOST_FRAMES:
                    ctrl.hand_lost()
                    cls.reset()
                    smoothed = None

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            draw_hud(frame, ctrl, fps, paused, hand_found)
            if debug:
                draw_debug(frame, cls)
            cv2.imshow("Hand Gesture Virtual Mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                ctrl.release_all()
                ctrl.notify("Paused" if paused else "Resumed")
            if key == ord("d"):
                debug = not debug

    ctrl.release_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
