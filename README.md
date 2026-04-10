# Mini_project17
#  Vision-Based Touchless Human–Computer Interaction System

> **Team 17 | Mini_project17**  
> Rudra Pratap Singh Rana · Utkarsh Pathak · Krish Gupta

Control your computer entirely through hand gestures — no physical contact required.  
Built with **MediaPipe**, **OpenCV**, **PyAutoGUI**, and a **Streamlit** web dashboard.

---

## 📸 Demo

| Gesture | Action |
|---------|--------|
|  Index finger only | Move cursor |
|  Index + Middle pinched | Left click |
|  Thumb + Index spread | Volume control |
|  Index + Middle (hand up/down) | Scroll up / down |

---

##  Project Structure

```
Mini_project17/
├── app.py                  # Streamlit web dashboard (main entry point)
├── requirements.txt        # Python dependencies
├── conftest.py             # Pytest configuration
├── modules/
│   ├── __init__.py
│   ├── hand_tracking.py    # MediaPipe hand detection & landmark extraction
│   ├── gesture_control.py  # Cursor movement & click detection
│   ├── volume_control.py   # Cross-platform system volume control
│   └── scroll_control.py   # Scroll gesture handler
└── tests/
    └── test_hand_tracking.py  # Full unit test suite (pytest)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rudra-pratap-singh-rana/Mini_project17.git
cd Mini_project17
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Windows users:** For hardware volume control, also run:
> ```bash
> pip install pycaw comtypes
> ```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.  
Click **Start** in the video widget, allow camera access, and begin gesturing.

---

## 🤚 Gesture Guide

| Gesture | Fingers Up | Action |
|---------|-----------|--------|
| Point   | Index only | **Move cursor** — hand position maps to screen |
| Click   | Index + Middle (pinch) | **Left click** — bring fingertips together |
| Volume  | Thumb + Index only | **Adjust volume** — spread = louder, close = quieter |
| Scroll Up | Index + Middle (hand in upper half) | **Scroll up** |
| Scroll Down | Index + Middle (hand in lower half) | **Scroll down** |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_hand_tracking.py::TestHandTrackerFingers::test_all_fingers_down_returns_five_zeros  PASSED
tests/test_hand_tracking.py::TestHandTrackerFingers::test_empty_landmarks_returns_all_zeros    PASSED
...
14 passed in 0.85s
```

---

## 🧩 Module Overview

### `modules/hand_tracking.py` — `HandTracker`
- Wraps MediaPipe Hands for detection and landmark extraction
- `find_hands(frame)` — detects and draws hand landmarks
- `get_landmarks(frame)` — returns `[(id, x, y), ...]` in pixel space
- `fingers_up(landmarks)` — returns `[thumb, index, middle, ring, pinky]` as 0/1
- `distance_between(p1, p2)` — Euclidean distance helper

### `modules/gesture_control.py` — `GestureController`
- `move_cursor(landmarks, fingers)` — maps index tip to screen coordinates with smoothening
- `detect_click(landmarks, fingers, tracker)` — click when index + middle tips converge

### `modules/volume_control.py` — `VolumeController`
- Factory function `get_volume_controller()` returns the right controller for your OS
- Supports **Windows** (pycaw), **macOS** (osascript), **Linux** (amixer)
- `distance_to_volume(dist)` — converts pixel distance to 0–100 volume level

### `modules/scroll_control.py` — `ScrollController`
- `process(landmarks, fingers, frame_h)` — triggers scroll based on hand vertical position

---

## 🔒 Security Notes

This is a local desktop application. If extending to a web service, consider:
- Adding authentication (e.g., JWT tokens)
- Input validation on all API endpoints
- Rate limiting on server routes
- CORS configuration for browser clients

---

## 🛣️ Roadmap

- [ ] Right-click gesture
- [ ] Double-click gesture  
- [ ] Drag-and-drop support
- [ ] Multi-hand gesture support
- [ ] Custom gesture trainer
- [ ] Gesture recording & playback

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-gesture`
3. Commit changes with descriptive messages: `git commit -m "feat: add right-click gesture"`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) — Hand landmark detection
- [OpenCV](https://opencv.org/) — Video capture and processing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — Cross-platform GUI automation
- [Streamlit](https://streamlit.io/) — Web dashboard framework
