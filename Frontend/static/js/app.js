/**
 * app.js
 * -------
 * Connects to the /status_stream SSE endpoint and updates all
 * dashboard UI elements in real time.
 */

"use strict";

// ── Module toggle state ───────────────────────────────────────────────────────

const moduleState = {
  enable_cursor: true,
  enable_click:  true,
  enable_volume: true,
  enable_scroll: true,
};

/**
 * Toggle a gesture module on/off and POST the update to the Flask backend.
 * @param {string} key  - Module key (e.g. 'enable_cursor')
 * @param {HTMLElement} el - The toggle label element
 */
window.toggleModule = async function (key, el) {
  moduleState[key] = !moduleState[key];
  const on = moduleState[key];

  el.classList.toggle("active", on);

  const swEl = el.querySelector(".tog-switch");
  swEl.textContent = on ? "ON" : "OFF";
  swEl.classList.toggle("on", on);

  try {
    await fetch("/settings", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ [key]: on }),
    });
  } catch (e) {
    console.warn("Settings update failed:", e);
  }
};

// ── Clock ─────────────────────────────────────────────────────────────────────

function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.toTimeString().slice(0, 8);
}
updateClock();
setInterval(updateClock, 1000);

// ── SSE Connection ────────────────────────────────────────────────────────────

const pill      = document.getElementById("connectionPill");
const pillLabel = document.getElementById("connectionLabel");

let actionFlash = null;

function connectSSE() {
  const es = new EventSource("/status_stream");

  es.onopen = () => {
    pill.classList.add("live");
    pillLabel.textContent = "LIVE";
  };

  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      updateUI(d);
    } catch (err) {
      console.error("SSE parse error:", err);
    }
  };

  es.onerror = () => {
    pill.classList.remove("live");
    pillLabel.textContent = "RECONNECTING";
    es.close();
    setTimeout(connectSSE, 2000);
  };
}

connectSSE();

// ── UI Updaters ───────────────────────────────────────────────────────────────

/**
 * Update all dashboard panels from a state snapshot object.
 * @param {object} d - State snapshot from the Flask backend
 */
function updateUI(d) {
  updateFPS(d.fps);
  updateHandStatus(d.hand_detected);
  updateFingers(d.fingers);
  updateAction(d.gesture_action);
  updateVolume(d.volume);
  updateFrameCount(d.frame_count);
}

function updateFPS(fps) {
  document.getElementById("fpsVal").textContent = fps;
}

function updateHandStatus(detected) {
  const icon  = document.getElementById("handIcon");
  const label = document.getElementById("handLabel");

  if (detected) {
    icon.classList.add("detected");
    label.classList.add("detected");
    label.textContent = "DETECTED";
  } else {
    icon.classList.remove("detected");
    label.classList.remove("detected");
    label.textContent = "SCANNING";
  }
}

function updateFingers(fingers) {
  fingers.forEach((up, i) => {
    const el = document.getElementById(`f${i}`);
    if (el) el.classList.toggle("up", !!up);
  });
}

function updateAction(text) {
  const el  = document.getElementById("actionText");
  const bar = document.getElementById("actionBar");

  const isActive = text && text !== "Waiting for gesture...";

  el.textContent = text || "Waiting for gesture...";
  el.classList.toggle("active", isActive);

  // Animate the action bar briefly when a gesture fires
  if (isActive) {
    if (actionFlash) clearTimeout(actionFlash);
    bar.style.width = "100%";
    actionFlash = setTimeout(() => { bar.style.width = "0%"; }, 400);
  }
}

function updateVolume(vol) {
  document.getElementById("volFill").style.width = `${vol}%`;
  document.getElementById("volPct").textContent  = `${vol}%`;
}

function updateFrameCount(count) {
  document.getElementById("frameCount").textContent = `FRAME ${count.toLocaleString()}`;
}
