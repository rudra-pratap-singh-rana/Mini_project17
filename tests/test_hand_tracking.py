"""
tests/test_hand_tracking.py
---------------------------
Unit tests for HandTracker, GestureController, VolumeController, ScrollController.
Run with:  pytest tests/ -v
"""

import math
import pytest
import numpy as np

# ─── HandTracker tests ────────────────────────────────────────────────────────

from modules.hand_tracking import HandTracker


@pytest.fixture
def tracker():
    return HandTracker()


def make_landmarks(positions: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Helper: build a 21-point landmark list from a list of (x, y) pairs."""
    return [(i, x, y) for i, (x, y) in enumerate(positions)]


def full_landmarks(default_x=300, default_y=300) -> list[tuple[int, int, int]]:
    """Return 21 default landmarks (all at the same point, used for shape checks)."""
    return [(i, default_x, default_y) for i in range(21)]


class TestHandTrackerFingers:

    def test_all_fingers_down_returns_five_zeros(self, tracker):
        lms = full_landmarks()
        # Tips (4,8,12,16,20) placed below their joints — fingers "down"
        result = tracker.fingers_up(lms)
        assert len(result) == 5

    def test_empty_landmarks_returns_all_zeros(self, tracker):
        assert tracker.fingers_up([]) == [0, 0, 0, 0, 0]

    def test_index_finger_up(self, tracker):
        """
        Index finger tip (id=8) should be above its joint (id=6) to count as up.
        y increases downward, so tip_y < joint_y means 'up'.
        """
        lms = full_landmarks(300, 300)
        lms_list = list(lms)
        # Set index tip (8) higher than index base (6)
        lms_list[8]  = (8, 300, 100)   # tip — high up
        lms_list[6]  = (6, 300, 200)   # base — lower
        result = tracker.fingers_up(lms_list)
        assert result[1] == 1, "Index finger should be detected as up"

    def test_distance_between_same_points_is_zero(self, tracker):
        p = (0, 100, 100)
        assert tracker.distance_between(p, p) == pytest.approx(0.0)

    def test_distance_between_known_points(self, tracker):
        p1 = (0, 0, 0)
        p2 = (1, 3, 4)
        assert tracker.distance_between(p1, p2) == pytest.approx(5.0)

    def test_distance_symmetric(self, tracker):
        p1 = (0, 10, 20)
        p2 = (1, 40, 60)
        assert tracker.distance_between(p1, p2) == pytest.approx(tracker.distance_between(p2, p1))


# ─── GestureController tests ─────────────────────────────────────────────────

from modules.gesture_control import GestureController


@pytest.fixture
def gesture():
    return GestureController(frame_w=640, frame_h=480)


class TestGestureController:

    def test_no_landmarks_returns_no_hand(self, gesture):
        result = gesture.move_cursor([], [0, 0, 0, 0, 0])
        assert result == "No hand detected"

    def test_idle_when_index_not_up(self, gesture):
        lms = full_landmarks()
        fingers = [0, 0, 0, 0, 0]  # no fingers up
        result = gesture.move_cursor(lms, fingers)
        assert result == "Idle"

    def test_moving_cursor_when_only_index_up(self, gesture, mocker):
        mocker.patch("pyautogui.moveTo")
        lms = full_landmarks(320, 240)
        fingers = [0, 1, 0, 0, 0]  # only index up
        result = gesture.move_cursor(lms, fingers)
        assert result == "Moving cursor"

    def test_click_not_triggered_when_fingers_apart(self, gesture, mocker):
        mocker.patch("pyautogui.click")
        tracker = HandTracker()
        lms = full_landmarks()
        # Index tip and middle tip far apart (> 35px)
        lms[8]  = (8, 100, 100)
        lms[12] = (12, 200, 200)
        fingers = [0, 1, 1, 0, 0]
        result = gesture.detect_click(lms, fingers, tracker)
        assert result == "No click"


# ─── VolumeController tests ───────────────────────────────────────────────────

from modules.volume_control import VolumeController, LinuxVolumeController


class ConcreteVolume(VolumeController):
    """Concrete subclass for testing the base class distance_to_volume method."""
    def set_volume(self, level):
        pass


class TestVolumeController:

    def test_min_distance_gives_zero_volume(self):
        vc = ConcreteVolume()
        assert vc.distance_to_volume(vc.MIN_DIST) == 0

    def test_max_distance_gives_full_volume(self):
        vc = ConcreteVolume()
        assert vc.distance_to_volume(vc.MAX_DIST) == 100

    def test_mid_distance_gives_mid_volume(self):
        vc = ConcreteVolume()
        mid = (vc.MIN_DIST + vc.MAX_DIST) / 2
        vol = vc.distance_to_volume(mid)
        assert 45 <= vol <= 55

    def test_below_min_clipped_to_zero(self):
        vc = ConcreteVolume()
        assert vc.distance_to_volume(0) == 0

    def test_above_max_clipped_to_hundred(self):
        vc = ConcreteVolume()
        assert vc.distance_to_volume(999) == 100


# ─── ScrollController tests ───────────────────────────────────────────────────

from modules.scroll_control import ScrollController


@pytest.fixture
def scroller():
    sc = ScrollController()
    sc._last_scroll_time = 0  # reset cooldown for tests
    return sc


class TestScrollController:

    def test_no_landmarks_returns_idle(self, scroller):
        result = scroller.process([], [0, 0, 0, 0, 0], 480)
        assert result == "Idle"

    def test_wrong_gesture_returns_idle(self, scroller):
        lms = full_landmarks()
        fingers = [1, 1, 1, 1, 1]  # all up — not scroll gesture
        result = scroller.process(lms, fingers, 480)
        assert result == "Idle"

    def test_scroll_up_when_hand_high(self, scroller, mocker):
        mocker.patch("pyautogui.scroll")
        import pyautogui
        lms = full_landmarks(300, 50)   # hand near top
        lms[8] = (8, 300, 50)          # index tip near top
        fingers = [0, 1, 1, 0, 0]
        result = scroller.process(lms, fingers, 480)
        assert result == "Scrolling Up"
        pyautogui.scroll.assert_called_once()

    def test_scroll_down_when_hand_low(self, scroller, mocker):
        mocker.patch("pyautogui.scroll")
        lms = full_landmarks(300, 440)
        lms[8] = (8, 300, 440)          # index tip near bottom
        fingers = [0, 1, 1, 0, 0]
        result = scroller.process(lms, fingers, 480)
        assert result == "Scrolling Down"
