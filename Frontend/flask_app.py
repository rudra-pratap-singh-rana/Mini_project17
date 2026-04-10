"""
flask_app.py
------------
Flask backend for the Vision-Based Touchless HCI System.

Routes:
  GET /               — Main dashboard UI
  GET /video_feed     — MJPEG webcam stream with gesture overlays
  GET /status_stream  — Server-Sent Events for real-time gesture status
  POST /settings      — Update gesture settings at runtime

Run with:
  python flask_app.py
"""

import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

from modules.hand_tracking import HandTracker
from modules.gesture_control import GestureController
from modules.volume_control import get_volume_controller
from modules.scroll_control import ScrollController

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── Shared State (thread-safe) ───────────────────────────────────────────────

class AppState:
    """Holds live gesture data shared between the video thread and HTTP routes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.gesture_action  = "Waiting..."
        self.hand_detected   = False
        self.fingers         = [0, 0, 0, 0, 0]
        self.volume          = 50
        self.fps             = 0
        self.frame_count     = 0

        # Feature toggles
        self.enable_cursor = True
        self.enable_click  = True
        self.enable_volume = True
        self.enable_scroll = True

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "gesture_action": self.gesture_action,
                "hand_detected":  self.hand_detected,
                "fingers":        list(self.fingers),
                "volume":         self.volume,
                "fps":            self.fps,
                "frame_count":    self.frame_count,
                "enable_cursor":  self.enable_cursor,
                "enable_click":   self.enable_click,
                "enable_volume":  self.enable_volume,
                "enable_scroll":  self.enable_scroll,
            }


state = AppState()

# ─── Camera & Gesture Thread ──────────────────────────────────────────────────

class Camera:
    """Captures frames, runs gesture recognition, and yields annotated MJPEG frames."""

    def __init__(self):
        self.cap      = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.tracker  = HandTracker(detection_conf=0.75, tracking_conf=0.75)
        self.gesture  = GestureController(frame_w=640, frame_h=480, smoothening=5)
        self.volume   = get_volume_controller()
        self.scroller = ScrollController()

        self.fps_timer = time.time()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def _overlay_hud(self, frame: np.ndarray, fingers: list, action: str, fps: int, vol: int):
        """Draw HUD elements — finger indicators, action bar, FPS."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (8, 8, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS {fps}", (w - 90, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 170), 2)

        # Title
        cv2.putText(frame, "TOUCHFREE HCI", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 170), 2)

        # Finger indicators (bottom-left)
        names  = ["T", "I", "M", "R", "P"]
        colors_on  = (0, 230, 160)
        colors_off = (50, 50, 70)
        for i, (name, up) in enumerate(zip(names, fingers)):
            cx = 20 + i * 36
            cy = h - 60
            cv2.circle(frame, (cx, cy), 13, colors_on if up else colors_off, -1)
            cv2.circle(frame, (cx, cy), 13, (0, 210, 170) if up else (80, 80, 100), 2)
            cv2.putText(frame, name, (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Action bar (bottom)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 36), (w, h), (8, 8, 20), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, action, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 160), 2)

        # Volume bar (right side)
        bx, by = w - 28, h - 60
        bar_h = int(np.interp(vol, [0, 100], [0, 120]))
        cv2.rectangle(frame, (bx, by), (bx + 14, by - 120), (30, 30, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + 14, by - bar_h), (0, 210, 170), -1)
        cv2.putText(frame, f"{vol}%", (bx - 10, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1)

        return frame

    def generate(self):
        """Yield MJPEG frames as a generator."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # FPS
            fps = int(1 / (time.time() - self.fps_timer + 1e-9))
            self.fps_timer = time.time()

            # Hand tracking
            frame = self.tracker.find_hands(frame)
            lms   = self.tracker.get_landmarks(frame)
            fingers = self.tracker.fingers_up(lms) if lms else [0] * 5

            actions = []

            if lms:
                if state.enable_cursor:
                    s = self.gesture.move_cursor(lms, fingers)
                    if s not in ("Idle", "No hand detected"):
                        actions.append(s)

                if state.enable_click:
                    s = self.gesture.detect_click(lms, fingers, self.tracker)
                    if s == "Click!":
                        actions.append("🖱 Click!")

                if state.enable_volume and fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
                    dist = self.tracker.distance_between(lms[4], lms[8])
                    vol  = self.volume.distance_to_volume(dist)
                    self.volume.set_volume(vol)
                    actions.append(f"Volume {vol}%")
                    state.update(volume=vol)

                if state.enable_scroll:
                    s = self.scroller.process(lms, fingers, h)
                    if s != "Idle":
                        actions.append(s)

            action_text = " · ".join(actions) if actions else "Waiting for gesture..."

            state.update(
                gesture_action=action_text,
                hand_detected=bool(lms),
                fingers=fingers,
                fps=fps,
                frame_count=state.frame_count + 1,
            )

            frame = self._overlay_hud(frame, fingers, action_text, fps, state.volume)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )


camera = Camera()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        camera.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status_stream")
def status_stream():
    """
    Server-Sent Events endpoint — pushes gesture state to the browser ~20x/sec.
    """
    def event_stream():
        while True:
            data = json.dumps(state.snapshot())
            yield f"data: {data}\n\n"
            time.sleep(0.05)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/settings", methods=["POST"])
def update_settings():
    """Update feature toggles at runtime."""
    body = request.get_json(force=True)
    valid_keys = {"enable_cursor", "enable_click", "enable_volume", "enable_scroll"}
    updates = {k: bool(v) for k, v in body.items() if k in valid_keys}
    state.update(**updates)
    return jsonify({"ok": True, "updated": updates})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🖐️  TouchFree HCI Dashboard")
    print("   Open → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
