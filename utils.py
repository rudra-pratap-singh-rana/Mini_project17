# utils.py
# Utility functions: distance calculation, finger state detection, smoothing

import math
import numpy as np


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.
    Each point is a tuple or list: (x, y)
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def fingers_up(landmarks, handedness="Right"):
    """
    Detect which fingers are extended (pointing up).

    MediaPipe hand landmarks:
      - Thumb tip: 4,  MCP: 2
      - Index tip: 8,  PIP: 6
      - Middle tip: 12, PIP: 10
      - Ring tip: 16,  PIP: 14
      - Pinky tip: 20, PIP: 18

    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    True = finger is up/extended
    """
    fingers = []

    # ----- Thumb -----
    # Thumb is extended if its tip is to the RIGHT of its IP joint (for right hand)
    # and to the LEFT for left hand (due to mirroring)
    tip = landmarks[4]
    ip  = landmarks[3]
    if handedness == "Right":
        fingers.append(tip[0] < ip[0])   # tip is left of IP in mirrored webcam
    else:
        fingers.append(tip[0] > ip[0])

    # ----- Four Fingers -----
    # A finger is extended if its tip y-coordinate is ABOVE (smaller y) its PIP joint
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip_id, pip_id in zip(tip_ids, pip_ids):
        fingers.append(landmarks[tip_id][1] < landmarks[pip_id][1])

    return fingers  # [thumb, index, middle, ring, pinky]


def smooth_value(current, previous, smoothing=0.5):
    """
    Simple exponential smoothing between current and previous value.
    smoothing=0.0 → no smoothing (use current directly)
    smoothing=0.9 → very heavy smoothing (very slow response)
    A value of 0.5 is a good default balance.
    """
    if previous is None:
        return current
    return previous * smoothing + current * (1 - smoothing)


def map_range(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another.
    Clamps output to [out_min, out_max].
    """
    # Prevent division by zero
    if in_max == in_min:
        return out_min
    mapped = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    return max(out_min, min(out_max, mapped))
