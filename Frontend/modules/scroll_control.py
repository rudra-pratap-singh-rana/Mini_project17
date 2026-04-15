"""
scroll_control.py
-----------------
Handles scroll gestures — two fingers up triggers scroll based on hand position.
"""

import pyautogui
import time


class ScrollController:
    """Scrolls the screen based on vertical hand position when two fingers are raised."""

    def __init__(self, scroll_speed: int = 3, cooldown: float = 0.08):
        """
        Initialize ScrollController.

        Args:
            scroll_speed: Number of scroll units per gesture tick.
            cooldown: Minimum seconds between scroll events.
        """
        self.scroll_speed = scroll_speed
        self.cooldown = cooldown
        self._last_scroll_time = 0
        self._reference_y: int | None = None

    def process(self, landmarks: list, fingers: list[int], frame_h: int) -> str:
        """
        Trigger scroll when index + middle fingers are up (and others down).

        Args:
            landmarks: List of (id, x, y) from HandTracker.get_landmarks().
            fingers: Output of HandTracker.fingers_up().
            frame_h: Frame height in pixels (used for mid-point calculation).

        Returns:
            Status string: 'Scrolling Up', 'Scrolling Down', or 'Idle'.
        """
        if not landmarks:
            return "Idle"

        # Two-finger scroll mode: index + middle up, ring + pinky down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            mid_y = landmarks[8][2]  # index tip y
            now = time.time()

            if now - self._last_scroll_time < self.cooldown:
                return "Idle"

            mid_frame = frame_h // 2
            if mid_y < mid_frame - 30:
                pyautogui.scroll(self.scroll_speed)
                self._last_scroll_time = now
                return "Scrolling Up"
            elif mid_y > mid_frame + 30:
                pyautogui.scroll(-self.scroll_speed)
                self._last_scroll_time = now
                return "Scrolling Down"

        return "Idle"
