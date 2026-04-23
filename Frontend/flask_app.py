"""
flask_app.py
------------
Flask backend for the Vision-Based Touchless HCI System.

Routes:
  GET  /              — Main dashboard UI
  GET  /video_feed    — MJPEG webcam stream with gesture overlays
  GET  /status_stream — Server-Sent Events for real-time gesture status
  POST /settings      — Update gesture settings at runtime
  GET  /health        — Health check endpoint

Run with:
  python flask_app.py
"""

import time
import json
import logging
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from modules.hand_tracking import HandTracker
from modules.gesture_control import GestureController
from modules.volume_control import get_volume_controller
from modules.scroll_control import ScrollController

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "touchfree-hci-team17-secret"

# ─── Rate Limiting ────────────────────────────────────────────────────────────

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute", "50 per second"],
    storage_uri="memory://",
)

# ─── Shared State (thread-safe) ───────────────────────────────────────────────

class AppState:
    """Holds live gesture data shared between the video thread and HTTP routes."""

    def __init__(self):
        self.lock            = threading.Lock()
        self.gesture_action  = "Waiting..."
        self.hand_detected   = False
        self.fingers         = [0, 0, 0, 0, 0]
        self.volume          = 50
        self.fps             = 0
        self.frame_count     = 0
        self.camera_error    = False

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
                "camera_error":   self.camera_error,
                "enable_cursor":  self.enable_cursor,
                "enable_click":   self.enable_click,
                "enable_volume":  self.enable_volume,
                "enable_scroll":  self.enable_scroll,
            }


state = AppState()

# ─── Allowed Settings Keys (input validation) ─────────────────────────────────

ALLOWED_SETTINGS = {"enable_cursor", "enable_click", "enable_volume", "enable_scroll"}

# ─── Camera & Gesture Thread ──────────────────────────────────────────────────

class Camera:
    """Captures frames, runs gesture recognition, and yields annotated MJPEG frames."""

    def __init__(self):
        self._init_camera()
        self._init_modules()
        self.fps_timer = time.time()

    def _init_camera(self):
        """Initialize webcam with error handling."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam on index 0.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            log.info("Camera initialized successfully.")
        except Exception as e:
            log.error(f"Camera init failed: {e}")
            self.cap = None
            state.update(camera_error=True)

    def _init_modules(self):
        """Initialize gesture modules with error handling."""
        try:
            self.tracker  = HandTracker(detection_conf=0.75, tracking_conf=0.75)
            self.gesture  = GestureController(frame_w=640, frame_h=480, smoothening=5)
            self.volume   = get_volume_controller()
            self.scroller = ScrollController()
            log.info("Gesture modules initialized successfully.")
        except Exception as e:
            log.error(f"Module init failed: {e}")
            self.tracker  = None
            self.gesture  = None
            self.volume   = None
            self.scroller = None

    def __del__(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                log.info("Camera released.")
        except Exception as e:
            log.warning(f"Error releasing camera: {e}")

    def _error_frame(self, message: str = "Camera Error") -> bytes:
        """Generate a black error frame with a text message."""
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, message, (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 60), 2)
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        except Exception:
            return b""

    def _overlay_hud(self, frame: np.ndarray, fingers: list, action: str, fps: int, vol: int):
        """Draw HUD elements — finger indicators, action bar, FPS."""
        try:
            h, w = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (8, 8, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, f"FPS {fps}", (w - 90, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 170), 2)
            cv2.putText(frame, "TOUCHFREE HCI", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 170), 2)

            names     = ["T", "I", "M", "R", "P"]
            color_on  = (0, 230, 160)
            color_off = (50, 50, 70)
            for i, (name, up) in enumerate(zip(names, fingers)):
                cx, cy = 20 + i * 36, h - 60
                cv2.circle(frame, (cx, cy), 13, color_on if up else color_off, -1)
                cv2.circle(frame, (cx, cy), 13, (0, 210, 170) if up else (80, 80, 100), 2)
                cv2.putText(frame, name, (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 36), (w, h), (8, 8, 20), -1)
            cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, action, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 160), 2)

            bx, by = w - 28, h - 60
            bar_h = int(np.interp(vol, [0, 100], [0, 120]))
            cv2.rectangle(frame, (bx, by), (bx + 14, by - 120), (30, 30, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + 14, by - bar_h), (0, 210, 170), -1)
            cv2.putText(frame, f"{vol}%", (bx - 10, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1)
        except Exception as e:
            log.warning(f"HUD overlay error: {e}")

        return frame

    def generate(self):
        """Yield MJPEG frames as a generator, with per-frame error handling."""
        if self.cap is None or not self.cap.isOpened():
            log.error("Camera not available — yielding error frames.")
            while True:
                frame_bytes = self._error_frame("No Camera Found — Check Connection")
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
                time.sleep(0.1)
            return

        consecutive_failures = 0
        MAX_FAILURES = 10

        while True:
            try:
                ok, frame = self.cap.read()

                if not ok:
                    consecutive_failures += 1
                    log.warning(f"Frame read failed ({consecutive_failures}/{MAX_FAILURES})")
                    if consecutive_failures >= MAX_FAILURES:
                        log.error("Too many consecutive frame failures — reinitializing camera.")
                        self._init_camera()
                        consecutive_failures = 0
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                fps = int(1 / (time.time() - self.fps_timer + 1e-9))
                self.fps_timer = time.time()

                fingers = [0] * 5
                lms     = []

                if self.tracker is not None:
                    try:
                        frame   = self.tracker.find_hands(frame)
                        lms     = self.tracker.get_landmarks(frame)
                        fingers = self.tracker.fingers_up(lms) if lms else [0] * 5
                    except Exception as e:
                        log.warning(f"Hand tracking error: {e}")

                actions = []

                if lms:
                    if state.enable_cursor and self.gesture:
                        try:
                            s = self.gesture.move_cursor(lms, fingers)
                            if s not in ("Idle", "No hand detected"):
                                actions.append(s)
                        except Exception as e:
                            log.warning(f"Cursor control error: {e}")

                    if state.enable_click and self.gesture:
                        try:
                            s = self.gesture.detect_click(lms, fingers, self.tracker)
                            if s == "Click!":
                                actions.append("Click!")
                        except Exception as e:
                            log.warning(f"Click detection error: {e}")

                    if (state.enable_volume and self.volume
                            and fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0):
                        try:
                            dist = self.tracker.distance_between(lms[4], lms[8])
                            vol  = self.volume.distance_to_volume(dist)
                            self.volume.set_volume(vol)
                            actions.append(f"Volume {vol}%")
                            state.update(volume=vol)
                        except Exception as e:
                            log.warning(f"Volume control error: {e}")

                    if state.enable_scroll and self.scroller:
                        try:
                            s = self.scroller.process(lms, fingers, h)
                            if s != "Idle":
                                actions.append(s)
                        except Exception as e:
                            log.warning(f"Scroll control error: {e}")

                action_text = " · ".join(actions) if actions else "Waiting for gesture..."

                state.update(
                    gesture_action=action_text,
                    hand_detected=bool(lms),
                    fingers=fingers,
                    fps=fps,
                    frame_count=state.frame_count + 1,
                    camera_error=False,
                )

                frame = self._overlay_hud(frame, fingers, action_text, fps, state.volume)

                try:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes() + b"\r\n"
                    )
                except Exception as e:
                    log.warning(f"Frame encode error: {e}")
                    continue

            except Exception as e:
                log.error(f"Unexpected error in generate loop: {e}")
                time.sleep(0.1)


try:
    camera = Camera()
    log.info("Camera object created.")
except Exception as e:
    log.critical(f"Fatal error creating Camera: {e}")
    camera = None

# ─── Global Error Handlers ────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405

@app.errorhandler(429)
def rate_limit_exceeded(e):
    log.warning(f"Rate limit exceeded from {get_remote_address()}")
    return jsonify({"error": "Too many requests — slow down.", "code": 429}), 429

@app.errorhandler(500)
def internal_error(e):
    log.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error", "code": 500}), 500

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
@limiter.limit("60 per minute")
def index():
    """Serve the main dashboard."""
    try:
        return render_template("index.html")
    except Exception as e:
        log.error(f"Failed to render index: {e}")
        abort(500)


@app.route("/video_feed")
@limiter.limit("10 per minute")
def video_feed():
    """MJPEG video stream endpoint."""
    if camera is None:
        log.error("video_feed called but camera is None.")
        abort(503)
    try:
        return Response(
            camera.generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        log.error(f"video_feed error: {e}")
        abort(500)


@app.route("/status_stream")
@limiter.limit("30 per minute")
def status_stream():
    """Server-Sent Events — pushes gesture state to the browser ~20x/sec."""
    def event_stream():
        while True:
            try:
                data = json.dumps(state.snapshot())
                yield f"data: {data}\n\n"
            except Exception as e:
                log.warning(f"SSE stream error: {e}")
                yield f"data: {json.dumps({'error': 'stream error'})}\n\n"
            time.sleep(0.05)

    try:
        return Response(event_stream(), mimetype="text/event-stream")
    except Exception as e:
        log.error(f"status_stream setup error: {e}")
        abort(500)


@app.route("/settings", methods=["POST"])
@limiter.limit("30 per minute")
def update_settings():
    """
    Update feature toggles at runtime.
    Validates that only known boolean keys are accepted.
    """
    try:
        body = request.get_json(force=True, silent=True)

        if body is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        if not isinstance(body, dict):
            return jsonify({"error": "Body must be a JSON object"}), 400

        updates = {}
        for k, v in body.items():
            if k not in ALLOWED_SETTINGS:
                return jsonify({"error": f"Unknown setting: '{k}'"}), 400
            if not isinstance(v, bool):
                return jsonify({"error": f"Value for '{k}' must be true or false"}), 400
            updates[k] = v

        if not updates:
            return jsonify({"error": "No valid settings provided"}), 400

        state.update(**updates)
        log.info(f"Settings updated: {updates}")
        return jsonify({"ok": True, "updated": updates})

    except Exception as e:
        log.error(f"Settings update error: {e}")
        return jsonify({"error": "Failed to update settings"}), 500


@app.route("/health")
@limiter.limit("120 per minute")
def health():
    """Health check — returns camera and system status."""
    try:
        cam_ok = (
            camera is not None
            and camera.cap is not None
            and camera.cap.isOpened()
        )
        return jsonify({
            "status":        "ok" if cam_ok else "degraded",
            "camera":        "ok" if cam_ok else "error",
            "hand_detected": state.hand_detected,
            "fps":           state.fps,
            "frame_count":   state.frame_count,
        })
    except Exception as e:
        log.error(f"Health check error: {e}")
        return jsonify({"status": "error", "detail": str(e)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🖐️  TouchFree HCI Dashboard")
    print("   Open   → http://localhost:5000")
    print("   Health → http://localhost:5000/health\n")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except Exception as e:
        log.critical(f"Failed to start Flask server: {e}")
