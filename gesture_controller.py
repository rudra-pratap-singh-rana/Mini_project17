# gesture_controller.py
# Converts hand landmarks + finger states into system actions:
#   - Cursor movement
#   - Left click (pinch)
#   - Scroll (two fingers vertical)
#   - Volume control (thumb + index spread)

import time
import pyautogui
import numpy as np

from utils import calculate_distance, fingers_up, smooth_value, map_range

# ── Optional: volume control via pycaw (Windows only) ──────────────────────────
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX = _volume_ctrl.GetVolumeRange()[:2]   # typically -65.25 to 0.0
    VOLUME_AVAILABLE = True
except Exception:
    VOLUME_AVAILABLE = False
    print("[INFO] pycaw not available — volume control disabled.")
# ───────────────────────────────────────────────────────────────────────────────

# Disable PyAutoGUI's built-in fail-safe pause (we add our own debounce)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # Move mouse to top-left corner to abort


class GestureController:
    """
    Maps detected hand landmarks to OS-level actions.

    Call update(landmarks, handedness, frame_w, frame_h) every frame.
    """

    # ── Tuning constants ────────────────────────────────────────────────────────
    SMOOTHING         = 0.60    # Cursor smoothing factor (0=none, 0.9=very smooth)
    CLICK_THRESHOLD   = 40      # Pixels: pinch distance to trigger click
    SCROLL_THRESHOLD  = 20      # Pixels: min vertical movement to scroll
    SCROLL_SPEED      = 3       # Scroll lines per trigger
    CLICK_COOLDOWN    = 0.4     # Seconds between allowed clicks
    SCROLL_COOLDOWN   = 0.08    # Seconds between scroll events
    VOL_COOLDOWN      = 0.05    # Seconds between volume updates
    # ───────────────────────────────────────────────────────────────────────────

    def __init__(self):
        # Screen resolution
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothed cursor coordinates (start at screen centre)
        self.smooth_x = self.screen_w  // 2
        self.smooth_y = self.screen_h // 2

        # Debounce timers
        self._last_click_time  = 0
        self._last_scroll_time = 0
        self._last_vol_time    = 0

        # Previous scroll y-position for delta calculation
        self._prev_scroll_y = None

        # State flags
        self._clicking = False   # True while pinch is held

        # Status message for on-screen display
        self.status = "No Hand"

    def update(self, landmarks, handedness, frame_w, frame_h):
        """
        Main entry point called every frame.

        landmarks  : list of 21 (x, y) pixel tuples
        handedness : "Right" or "Left"
        frame_w/h  : camera frame dimensions (used for coordinate mapping)
        """
        if landmarks is None:
            self.status = "No Hand Detected"
            return

        # Detect which fingers are extended
        up = fingers_up(landmarks, handedness)
        # up = [thumb, index, middle, ring, pinky]
        thumb, index, middle, ring, pinky = up

        # Key landmark positions
        index_tip  = landmarks[8]    # Index finger tip
        thumb_tip  = landmarks[4]    # Thumb tip
        middle_tip = landmarks[12]   # Middle finger tip

        # ── Gesture priority (most specific first) ──────────────────────────

        # 1. VOLUME CONTROL: thumb + index + middle all up, ring + pinky down
        #    Distance between thumb and index tip controls volume
        if thumb and index and middle and not ring and not pinky:
            self._handle_volume(thumb_tip, index_tip)
            self.status = "Volume Control 🔊"
            return

        # 2. SCROLL: index + middle up, others down — vertical movement scrolls
        if not thumb and index and middle and not ring and not pinky:
            self._handle_scroll(index_tip, middle_tip)
            self.status = "Scrolling ↕"
            return

        # 3. CLICK (PINCH): index up, thumb close to index tip
        if not thumb and index and not middle and not ring and not pinky:
            pinch_dist = calculate_distance(thumb_tip, index_tip)
            if pinch_dist < self.CLICK_THRESHOLD:
                self._handle_click()
                self.status = "Click! 🖱️"
            else:
                # 4. MOVE CURSOR: only index finger up
                self._handle_move(index_tip, frame_w, frame_h)
                self.status = "Moving Cursor 👆"
            return

        # 5. CURSOR MOVE (explicit): only index up (thumb down)
        if not thumb and index and not middle and not ring and not pinky:
            self._handle_move(index_tip, frame_w, frame_h)
            self.status = "Moving Cursor 👆"
            return

        # Default: show finger state
        self.status = f"{''.join(['👍' if thumb else '','☝' if index else '','🖕' if middle else '','💍' if ring else '','🤙' if pinky else ''])}"

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _handle_move(self, index_tip, frame_w, frame_h):
        """
        Map index finger tip position to screen coordinates with smoothing.
        We use a reduced active zone (10%-90% of frame) so the cursor reaches
        screen edges without needing to move the finger to the very edge.
        """
        # Map frame coordinates → screen coordinates
        # Shrink the input range to make edge-reaching easier
        margin_x = int(frame_w * 0.10)
        margin_y = int(frame_h * 0.10)

        raw_x = map_range(
            index_tip[0],
            margin_x, frame_w - margin_x,
            0, self.screen_w
        )
        raw_y = map_range(
            index_tip[1],
            margin_y, frame_h - margin_y,
            0, self.screen_h
        )

        # Apply exponential smoothing to reduce jitter
        self.smooth_x = smooth_value(raw_x, self.smooth_x, self.SMOOTHING)
        self.smooth_y = smooth_value(raw_y, self.smooth_y, self.SMOOTHING)

        # Move the actual mouse cursor
        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

    def _handle_click(self):
        """Trigger a left click with debounce to prevent repeated clicks."""
        now = time.time()
        if not self._clicking and (now - self._last_click_time) > self.CLICK_COOLDOWN:
            pyautogui.click()
            self._last_click_time = now
            self._clicking = True
        elif (now - self._last_click_time) > self.CLICK_COOLDOWN:
            self._clicking = False   # Reset so next pinch can click

    def _handle_scroll(self, index_tip, middle_tip):
        """
        Scroll based on the vertical movement of two fingers.
        Positive scroll = up, negative = down.
        """
        now = time.time()
        if (now - self._last_scroll_time) < self.SCROLL_COOLDOWN:
            return

        # Use midpoint of index and middle finger tips
        mid_y = (index_tip[1] + middle_tip[1]) // 2

        if self._prev_scroll_y is None:
            self._prev_scroll_y = mid_y
            return

        delta = self._prev_scroll_y - mid_y   # Positive = fingers moved up = scroll up

        if abs(delta) > self.SCROLL_THRESHOLD:
            direction = 1 if delta > 0 else -1
            pyautogui.scroll(direction * self.SCROLL_SPEED)
            self._prev_scroll_y = mid_y
            self._last_scroll_time = now
        else:
            self._prev_scroll_y = mid_y

    def _handle_volume(self, thumb_tip, index_tip):
        """
        Control system volume based on distance between thumb and index tip.
        Short distance = low volume, large distance = high volume.
        """
        if not VOLUME_AVAILABLE:
            return

        now = time.time()
        if (now - self._last_vol_time) < self.VOL_COOLDOWN:
            return

        dist = calculate_distance(thumb_tip, index_tip)

        # Map distance (roughly 20–200 px) to volume range
        vol = map_range(dist, 20, 200, VOL_MIN, VOL_MAX)
        try:
            _volume_ctrl.SetMasterVolumeLevel(vol, None)
        except Exception:
            pass

        self._last_vol_time = now
