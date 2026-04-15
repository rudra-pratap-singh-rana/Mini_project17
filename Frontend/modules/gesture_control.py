"""
gesture_control.py
------------------
Translates hand landmarks into cursor movements and click actions.
"""

import numpy as np
import pyautogui
import time

pyautogui.FAILSAFE = False


class GestureController:
    """Maps hand gestures to mouse cursor actions."""

    # Landmark indices
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    THUMB_TIP = 4

    def __init__(
        self,
        frame_w: int = 640,
        frame_h: int = 480,
        smoothening: int = 5,
        margin: int = 80,
    ):
        """
        Initialize the GestureController.

        Args:
            frame_w: Width of the camera frame in pixels.
            frame_h: Height of the camera frame in pixels.
            smoothening: Number of frames to smooth cursor movement over.
            margin: Pixel margin to shrink the active gesture area.
        """
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.margin = margin
        self.smoothening = smoothening

        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self._last_click_time = 0
        self.click_cooldown = 0.4  # seconds

    def move_cursor(self, landmarks: list, fingers: list[int]) -> str:
        """
        Move mouse cursor based on index finger position when only index is up.

        Args:
            landmarks: List of (id, x, y) tuples from HandTracker.
            fingers: Output of HandTracker.fingers_up().

        Returns:
            Status string describing the action taken.
        """
        if not landmarks:
            return "No hand detected"

        # Moving mode: only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            x1 = landmarks[self.INDEX_TIP][1]
            y1 = landmarks[self.INDEX_TIP][2]

            screen_x = np.interp(x1, [self.margin, self.frame_w - self.margin], [0, self.screen_w])
            screen_y = np.interp(y1, [self.margin, self.frame_h - self.margin], [0, self.screen_h])

            self.curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
            self.curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening

            pyautogui.moveTo(self.curr_x, self.curr_y)
            self.prev_x, self.prev_y = self.curr_x, self.curr_y
            return "Moving cursor"

        return "Idle"

    def detect_click(self, landmarks: list, fingers: list[int], tracker) -> str:
        """
        Perform a left-click when index and middle fingers are close together.

        Args:
            landmarks: List of (id, x, y) tuples from HandTracker.
            fingers: Output of HandTracker.fingers_up().
            tracker: HandTracker instance (for distance calculation).

        Returns:
            Status string.
        """
        if fingers[1] == 1 and fingers[2] == 1:
            dist = tracker.distance_between(
                landmarks[self.INDEX_TIP], landmarks[self.MIDDLE_TIP]
            )
            if dist < 35:
                now = time.time()
                if now - self._last_click_time > self.click_cooldown:
                    pyautogui.click()
                    self._last_click_time = now
                    return "Click!"
        return "No click"
