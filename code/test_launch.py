import os
import sys
from dotenv import load_dotenv

print("🔍 STEP 1: Testing environment loading...")
load_dotenv()
csv_path = os.getenv("OPENFACE_CSV_PATH", "openface_output.csv")
print(f"[OK] Target OpenFace path is set to: {csv_path}")
print(f"[CHECK] Does this file exist right now? -> {os.path.exists(csv_path)}")

print("\n🎙️ STEP 2: Testing Sound Device list...")
try:
    import sounddevice as sd
    # This will print out every microphone and speaker on your computer
    print(sd.query_devices())
    print(f"[OK] Default input device index is: {sd.default.device[0]}")
except Exception as e:
    print(f"[FAIL] Audio system check failed: {e}")

print("\n🚀 STEP 3: Initializing background monitors...")
try:
    from tilt_backend import STATE, BehaviorMonitor, VoiceMonitor, FacialMonitor
    print("[OK] All code objects loaded into memory successfully.")
except Exception as e:
    print(f"[FAIL] Teammate code import error: {e}")

print("\nDiagnostic complete. If you see this message, the core packages are fine.")
