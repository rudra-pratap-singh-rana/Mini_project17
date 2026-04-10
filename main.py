# main.py
# Entry point: captures webcam, runs hand tracking + gesture control,
# displays annotated feed with FPS and status overlay.

import sys
import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from gesture_controller import GestureController

# ── Configuration ───────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0       # 0 = default webcam; try 1 or 2 if this fails
FRAME_WIDTH    = 640     # Resize capture for performance
FRAME_HEIGHT   = 480
FLIP_FRAME     = True    # Mirror the frame (feels more natural)
# ────────────────────────────────────────────────────────────────────────────────


def draw_overlay(frame, fps: float, status: str):
    """
    Draw a semi-transparent HUD on the frame showing FPS and gesture status.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark banner at the top
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS counter (top-left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA)

    # Gesture status (top-centre)
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, status, (text_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Instruction footer
    footer = "Press Q to quit"
    cv2.putText(frame, footer, (w - 160, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    # ── Open webcam ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAMERA_INDEX}.")
        print("        Try changing CAMERA_INDEX to 1 or 2 in main.py")
        sys.exit(1)

    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Camera opened successfully.")
    print("[INFO] Starting gesture control — press Q in the window to quit.")
    print("[INFO] Move mouse to TOP-LEFT screen corner to emergency-stop PyAutoGUI.")

    # ── Initialise modules ───────────────────────────────────────────────────────
    tracker    = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)
    controller = GestureController()

    # FPS calculation
    prev_time = time.time()
    fps_smooth = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — retrying…")
            continue

        # Mirror frame so movements feel natural (like a mirror)
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

        # ── Hand detection ───────────────────────────────────────────────────────
        landmarks, handedness, frame = tracker.find_hands(frame)

        # ── Gesture → action ─────────────────────────────────────────────────────
        controller.update(landmarks, handedness, frame_w, frame_h)

        # ── FPS calculation (smoothed) ───────────────────────────────────────────
        now      = time.time()
        raw_fps  = 1.0 / max(now - prev_time, 1e-6)
        fps_smooth = fps_smooth * 0.8 + raw_fps * 0.2   # Smooth FPS display
        prev_time = now

        # ── Draw HUD ─────────────────────────────────────────────────────────────
        draw_overlay(frame, fps_smooth, controller.status)

        # ── Show window ──────────────────────────────────────────────────────────
        cv2.imshow("Gesture Control — Hand Tracker", frame)

        # Exit on Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit signal received.")
            break

    # ── Cleanup ──────────────────────────────────────────────────────────────────
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed cleanly.")


if __name__ == "__main__":
    main()
