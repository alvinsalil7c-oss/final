"""
TILT AI - Combined Stress Detection System
============================================
Combines three signal sources into one live stress dashboard:

  1. VOICE     - Hume AI vocal prosody analysis (live microphone -> Hume EVI websocket)
  2. BEHAVIOR  - Keyboard & mouse dynamics (typing speed, backspaces, clicks, mouse movement)
  3. FACIAL    - OpenFace Action Unit (AU) data, read live from a CSV that OpenFace writes
                 while running its own capture process (this script tails that file)

Run this script WHILE OpenFace's FeatureExtraction / OpenFaceOffline is running with
live CSV output enabled, e.g.:

    FeatureExtraction.exe -device 0 -of openface_output.csv

Then point OPENFACE_CSV_PATH below (or set the OPENFACE_CSV_PATH env var) at that file.
The facial monitor waits patiently for that file to appear, so you can start this script
before or after OpenFace - order doesn't matter.

Original standalone scripts this replaces (no feature lost, just reorganized):
  - test_hume.py          -> merged into check_hume_key() at startup
  - emotion_detector.py   -> merged into VoiceMonitor
  - behaviour_detector.py -> merged into BehaviorMonitor
  (voice.wav was a test asset, not read by any script directly - live mic input is unchanged)

Dependencies (same as before, nothing new except stdlib csv/collections):
    pip install sounddevice websockets python-dotenv pynput numpy
"""

import os
import time
import json
import base64
import asyncio
import urllib.parse
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import websockets
from pynput import keyboard, mouse
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

HUME_API_KEY = os.getenv("HUME_API_KEY")

MIC_SAMPLE_RATE = 44100
MIC_CHANNELS = 1
MIC_DEVICE = 1          # change to match your input device index
MIC_BLOCK_SIZE = 4410   # ~100ms

OPENFACE_CSV_PATH = os.getenv("OPENFACE_CSV_PATH", "openface_output.csv")
OPENFACE_POLL_SECONDS = 0.5
OPENFACE_WINDOW_SIZE = 100     # rolling frames used for facial stress score

DASHBOARD_INTERVAL_SECONDS = 10

# Hume prosody emotions treated as "stress-relevant" when computing a voice stress score
STRESS_EMOTIONS = [
    "Anxiety", "Distress", "Fear", "Tension", "Anger",
    "Horror", "Confusion", "Embarrassment", "Pain",
]

# OpenFace Action Units commonly associated with tension/stress expressions
STRESS_AUS = ["AU04_r", "AU07_r", "AU23_r", "AU20_r"]


# ============================================================
# STARTUP CHECK  (was test_hume.py)
# ============================================================

def check_hume_key():
    if HUME_API_KEY:
        print("[OK] Hume API key found.")
        return True
    print("[FAIL] HUME_API_KEY not found in .env - voice monitoring will be disabled.")
    return False


# ============================================================
# SHARED STATE
# ============================================================

class StressState:
    """Thread-safe container all three monitors write into and the dashboard reads from."""

    def __init__(self):
        self._lock = threading.Lock()

        # behavior
        self.key_times = []
        self.backspace_count = 0
        self.click_count = 0
        self.mouse_distances = []

        # voice
        self.voice_top_emotions = []   # most recent list of (name, score)
        self.voice_transcript = None

        # facial
        self.facial_frames = deque(maxlen=OPENFACE_WINDOW_SIZE)  # rolling AU dicts
        self.facial_available = False

    # ---- behavior writers ----
    def record_key_interval(self, interval):
        with self._lock:
            self.key_times.append(interval)

    def record_backspace(self):
        with self._lock:
            self.backspace_count += 1

    def record_click(self):
        with self._lock:
            self.click_count += 1

    def record_mouse_distance(self, dist):
        with self._lock:
            self.mouse_distances.append(dist)

    # ---- voice writer ----
    def update_voice(self, top_emotions, transcript=None):
        with self._lock:
            self.voice_top_emotions = top_emotions
            if transcript:
                self.voice_transcript = transcript

    # ---- facial writer ----
    def record_facial_frame(self, au_dict):
        with self._lock:
            self.facial_frames.append(au_dict)
            self.facial_available = True

    # ---- snapshot for the dashboard ----
    def snapshot(self):
        with self._lock:
            return {
                "key_times": list(self.key_times),
                "backspace_count": self.backspace_count,
                "click_count": self.click_count,
                "mouse_distances": list(self.mouse_distances),
                "voice_top_emotions": list(self.voice_top_emotions),
                "voice_transcript": self.voice_transcript,
                "facial_frames": list(self.facial_frames),
                "facial_available": self.facial_available,
            }


STATE = StressState()


# ============================================================
# BEHAVIOR MONITOR  (was behaviour-detector.py)
# ============================================================

class BehaviorMonitor:
    def __init__(self, state: StressState):
        self.state = state
        self.last_key_time = None
        self.last_mouse_position = None

    def on_press(self, key):
        now = time.time()
        if self.last_key_time is not None:
            self.state.record_key_interval(now - self.last_key_time)
        self.last_key_time = now

        try:
            if key == keyboard.Key.backspace:
                self.state.record_backspace()
        except Exception:
            pass

    def on_move(self, x, y):
        if self.last_mouse_position is not None:
            old_x, old_y = self.last_mouse_position
            distance = np.sqrt((x - old_x) ** 2 + (y - old_y) ** 2)
            self.state.record_mouse_distance(distance)
        self.last_mouse_position = (x, y)

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.state.record_click()

    def start(self):
        keyboard.Listener(on_press=self.on_press).start()
        mouse.Listener(on_move=self.on_move, on_click=self.on_click).start()
        print("[OK] Behavior monitor (keyboard/mouse) started.")

    @staticmethod
    def score(snapshot):
        key_times = snapshot["key_times"]
        mouse_distances = snapshot["mouse_distances"]

        avg_key_interval = np.mean(key_times) if key_times else 0
        typing_speed = (1 / avg_key_interval) if avg_key_interval else 0
        avg_mouse_movement = np.mean(mouse_distances) if mouse_distances else 0

        stress_score = 0
        if typing_speed > 5:
            stress_score += 30
        if snapshot["backspace_count"] > 10:
            stress_score += 20
        if snapshot["click_count"] > 30:
            stress_score += 20
        if avg_mouse_movement > 20:
            stress_score += 20
        stress_score = min(stress_score, 100)

        return {
            "score": stress_score,
            "typing_speed": round(typing_speed, 2),
            "backspaces": snapshot["backspace_count"],
            "clicks": snapshot["click_count"],
            "mouse_movement": round(avg_mouse_movement, 2),
        }


# ============================================================
# VOICE MONITOR  (was emotion_detector.py)
# ============================================================

class VoiceMonitor:
    def __init__(self, state: StressState, api_key: str):
        self.state = state
        self.api_key = api_key

    async def run(self):
        if not self.api_key:
            print("[SKIP] Voice monitor disabled - no HUME_API_KEY.")
            return

        encoded_key = urllib.parse.quote(self.api_key)
        url = (
            "wss://api.hume.ai/v0/evi/chat"
            f"?api_key={encoded_key}"
            "&verbose_transcription=true"
        )

        print("[..] Connecting to Hume...")

        try:
            async with websockets.connect(url) as websocket:
                print("[OK] Voice monitor connected to Hume.")

                await websocket.send(json.dumps({
                    "type": "session_settings",
                    "audio": {
                        "encoding": "linear16",
                        "sample_rate": MIC_SAMPLE_RATE,
                        "channels": MIC_CHANNELS,
                    },
                }))

                loop = asyncio.get_running_loop()

                def audio_callback(indata, frames, t, status):
                    if status:
                        print("Microphone status:", status)
                    audio_bytes = indata.tobytes()
                    message = {
                        "type": "audio_input",
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    }
                    asyncio.run_coroutine_threadsafe(
                        websocket.send(json.dumps(message)), loop
                    )

                with sd.InputStream(
                    samplerate=MIC_SAMPLE_RATE,
                    channels=MIC_CHANNELS,
                    dtype="int16",
                    blocksize=MIC_BLOCK_SIZE,
                    device=MIC_DEVICE,
                    callback=audio_callback,
                ):
                    async for raw_message in websocket:
                        data = json.loads(raw_message)
                        message_type = data.get("type")

                        if message_type == "error":
                            print("[HUME ERROR]", data)
                            continue

                        if message_type == "chat_metadata":
                            print("[OK] Hume chat started.")

                        if message_type == "user_message":
                            if data.get("interim"):
                                continue

                            transcript = data.get("message", {}).get("content", "")

                            scores = (
                                data.get("models", {})
                                .get("prosody", {})
                                .get("scores", {})
                            )

                            if scores:
                                top_emotions = sorted(
                                    scores.items(), key=lambda kv: kv[1], reverse=True
                                )[:5]
                                self.state.update_voice(top_emotions, transcript or None)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[!] Hume connection closed. Code={e.code} Reason={e.reason}")
        except Exception as e:
            print(f"[!] Voice monitor error: {type(e).__name__}: {e}")

    @staticmethod
    def score(snapshot):
        top_emotions = snapshot["voice_top_emotions"]
        if not top_emotions:
            return {"score": None, "top_emotions": []}

        relevant = [s for name, s in top_emotions if name in STRESS_EMOTIONS]
        voice_score = (
            round(min(sum(relevant) / max(len(relevant), 1) * 100, 100), 1)
            if relevant else 0.0
        )

        return {"score": voice_score, "top_emotions": top_emotions}


# ============================================================
# FACIAL MONITOR  (new - tails OpenFace's live CSV output)
# ============================================================

class FacialMonitor:
    """Tails a CSV file that OpenFace writes live (one row appended per processed frame)."""

    def __init__(self, state: StressState, csv_path: str):
        self.state = state
        self.csv_path = csv_path
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _wait_for_file(self):
        first = True
        while not self._stop.is_set() and not os.path.exists(self.csv_path):
            if first:
                print(f"[..] Facial monitor waiting for OpenFace CSV at: {self.csv_path}")
                first = False
            time.sleep(1)

    def run(self):
        self._wait_for_file()
        if self._stop.is_set():
            return

        try:
            with open(self.csv_path, "r", newline="") as f:
                header_line = f.readline()
                if not header_line:
                    return
                fieldnames = [h.strip() for h in header_line.strip().split(",")]
                print("[OK] Facial monitor attached to OpenFace CSV.")

                while not self._stop.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(OPENFACE_POLL_SECONDS)
                        continue

                    values = [v.strip() for v in line.strip().split(",")]
                    if len(values) != len(fieldnames):
                        continue  # malformed/partial row, skip

                    row = dict(zip(fieldnames, values))
                    au_dict = {}
                    for au in STRESS_AUS:
                        if au in row:
                            try:
                                au_dict[au] = float(row[au])
                            except ValueError:
                                pass

                    if au_dict:
                        self.state.record_facial_frame(au_dict)

        except FileNotFoundError:
            print(f"[!] OpenFace CSV disappeared: {self.csv_path}")
        except Exception as e:
            print(f"[!] Facial monitor error: {type(e).__name__}: {e}")

    @staticmethod
    def score(snapshot):
        frames = snapshot["facial_frames"]
        if not snapshot["facial_available"] or not frames:
            return {"score": None, "au_averages": {}}

        au_averages = {}
        for au in STRESS_AUS:
            vals = [f[au] for f in frames if au in f]
            au_averages[au] = round(float(np.mean(vals)), 2) if vals else 0.0

        # Simple threshold scoring, same additive style as the behavior score
        facial_score = 0
        if au_averages.get("AU04_r", 0) > 1.0:   # brow lowerer - tension/frown
            facial_score += 30
        if au_averages.get("AU07_r", 0) > 1.0:   # lid tightener - squint/stress
            facial_score += 25
        if au_averages.get("AU23_r", 0) > 1.0:   # lip tightener
            facial_score += 25
        if au_averages.get("AU20_r", 0) > 1.0:   # lip stretcher - fear/anxiety
            facial_score += 20
        facial_score = min(facial_score, 100)

        return {"score": facial_score, "au_averages": au_averages}


# ============================================================
# DASHBOARD  (combines all three signals)
# ============================================================

def classify(score):
    if score is None:
        return "N/A"
    if score < 30:
        return "Calm"
    elif score < 60:
        return "Moderate Stress"
    else:
        return "High Stress"


async def dashboard_loop():
    while True:
        await asyncio.sleep(DASHBOARD_INTERVAL_SECONDS)

        snapshot = STATE.snapshot()
        behavior = BehaviorMonitor.score(snapshot)
        voice = VoiceMonitor.score(snapshot)
        facial = FacialMonitor.score(snapshot)

        available_scores = [
            s for s in (behavior["score"], voice["score"], facial["score"])
            if s is not None
        ]
        combined_score = (
            round(sum(available_scores) / len(available_scores), 1)
            if available_scores else None
        )

        print("\n==================================================")
        print("            TILT AI - COMBINED STRESS DASHBOARD")
        print("==================================================")

        print(f"BEHAVIOR  : {behavior['score']:>5}%  ({classify(behavior['score'])})")
        print(f"            typing={behavior['typing_speed']}  backspaces={behavior['backspaces']}  "
              f"clicks={behavior['clicks']}  mouse_move={behavior['mouse_movement']}")

        if voice["score"] is not None:
            print(f"VOICE     : {voice['score']:>5}%  ({classify(voice['score'])})")
            top_str = ", ".join(f"{n}={v:.2f}" for n, v in voice["top_emotions"])
            print(f"            top emotions: {top_str}")
        else:
            print("VOICE     :   N/A  (no data yet)")

        if facial["score"] is not None:
            print(f"FACIAL    : {facial['score']:>5}%  ({classify(facial['score'])})")
            au_str = ", ".join(f"{k}={v}" for k, v in facial["au_averages"].items())
            print(f"            AUs: {au_str}")
        else:
            print("FACIAL    :   N/A  (waiting on OpenFace CSV)")

        print("--------------------------------------------------")
        if combined_score is not None:
            print(f"COMBINED  : {combined_score:>5}%  ({classify(combined_score)})")
        else:
            print("COMBINED  :   N/A  (no signals yet)")
        print("==================================================\n")


# ============================================================
# MAIN
# ============================================================

async def main():
    check_hume_key()

    BehaviorMonitor(STATE).start()

    facial_monitor = FacialMonitor(STATE, OPENFACE_CSV_PATH)
    threading.Thread(target=facial_monitor.run, daemon=True).start()

    voice_monitor = VoiceMonitor(STATE, HUME_API_KEY)

    print()
    print("TILT AI combined monitor running.")
    print("Speak into your mic, type/move normally, and keep OpenFace running for facial data.")
    print(f"Dashboard prints every {DASHBOARD_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")
    print()

    await asyncio.gather(
        voice_monitor.run(),
        dashboard_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
