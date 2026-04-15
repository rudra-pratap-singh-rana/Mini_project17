"""
hand_tracking.py
----------------
Handles hand detection and landmark extraction using MediaPipe.
"""

import cv2
import mediapipe as mp
import math


class HandTracker:
    """Detects and tracks hand landmarks from a video frame."""

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.7, tracking_conf: float = 0.7):
        """
        Initialize the HandTracker.

        Args:
            max_hands: Maximum number of hands to detect.
            detection_conf: Minimum detection confidence threshold.
            tracking_conf: Minimum tracking confidence threshold.
        """
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=2, circle_radius=2)

    def find_hands(self, frame: "np.ndarray", draw: bool = True) -> "np.ndarray":
        """
        Detect hands in a BGR frame and optionally draw landmarks.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to draw landmarks on the frame.

        Returns:
            Annotated frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.draw_spec,
                    self.draw_spec,
                )
        return frame

    def get_landmarks(self, frame: "np.ndarray", hand_index: int = 0) -> list[tuple[int, int, int]]:
        """
        Return pixel-space (id, x, y) landmarks for a given hand.

        Args:
            frame: The current video frame (used for dimensions).
            hand_index: Index of the hand (0 = first detected).

        Returns:
            List of (id, x, y) tuples, or empty list if no hand found.
        """
        landmarks = []
        if self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]
                h, w, _ = frame.shape
                for idx, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((idx, cx, cy))
        return landmarks

    def fingers_up(self, landmarks: list) -> list[int]:
        """
        Determine which fingers are extended.

        Args:
            landmarks: List of (id, x, y) from get_landmarks().

        Returns:
            List of 5 ints [thumb, index, middle, ring, pinky] — 1 if up, 0 if down.
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb: compare x-axis (tip vs joint)
        fingers.append(1 if landmarks[tips[0]][1] < landmarks[tips[0] - 1][1] else 0)

        # Other fingers: compare y-axis (tip vs two joints below tip)
        for tip in tips[1:]:
            fingers.append(1 if landmarks[tip][2] < landmarks[tip - 2][2] else 0)

        return fingers

    def distance_between(self, p1: tuple, p2: tuple) -> float:
        """
        Euclidean distance between two landmark points.

        Args:
            p1: (id, x, y) tuple.
            p2: (id, x, y) tuple.

        Returns:
            Float distance in pixels.
        """
        return math.hypot(p2[1] - p1[1], p2[2] - p1[2])
