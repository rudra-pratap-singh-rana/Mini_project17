# hand_tracker.py
# Handles MediaPipe hand detection and landmark extraction.

import cv2
import mediapipe as mp


class HandTracker:
    """
    Wraps MediaPipe Hands for easy landmark detection.

    Usage:
        tracker = HandTracker()
        landmarks, handedness = tracker.find_hands(frame)
    """

    def __init__(
        self,
        max_hands=1,            # Track only 1 hand for performance
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):
        # Initialize MediaPipe Hands solution
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # Video mode (uses tracking between frames)
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def find_hands(self, frame):
        """
        Process a BGR frame, detect hands, draw landmarks.

        Returns:
            landmarks   : list of 21 (x, y) tuples in pixel coords, or None
            handedness  : "Right" or "Left" string, or None
            frame       : annotated frame with landmarks drawn
        """
        # Convert BGR → RGB (MediaPipe requires RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance hint: mark image as not writeable to save a copy
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, None, frame

        # --- Use the first detected hand only ---
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        # Extract (x, y) pixel coordinates for all 21 landmarks
        landmarks = []
        for lm in hand_landmarks.landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            landmarks.append((cx, cy))

        # Get handedness label ("Right" / "Left")
        handedness = results.multi_handedness[0].classification[0].label

        # Draw the hand skeleton on the frame
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

        return landmarks, handedness, frame

    def release(self):
        """Free MediaPipe resources."""
        self.hands.close()
