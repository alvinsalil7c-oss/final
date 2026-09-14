import sys
import os
import asyncio
import threading
import time
import customtkinter as ctk

# ⚡ ENFORCE DIRECT PATH PARAMETERS
os.environ["OPENFACE_CSV_PATH"] = r"C:\hackathon\openface\openface_output.csv"
os.environ["HUME_API_KEY"] = "Oc1B5SKVRpBJu0ZVvGnAW96QWTH6tqrArZNIb3PBZMg3m3b1"

# Import your teammate's exact code structures
from tilt_backend import STATE, BehaviorMonitor, VoiceMonitor, FacialMonitor, main

# Initialize the CustomTkinter UI Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TiltDashboardUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("TILT AI - Live Stress Detection System")
        self.geometry("700x550")
        self.resizable(False, False)

        # --- TITLE ---
        self.title_label = ctk.CTkLabel(self, text="⚡ TILT AI STRESS MONITOR", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)

        # --- MAIN METRIC (COMBINED SCORE) ---
        self.score_frame = ctk.CTkFrame(self, width=600, height=120)
        self.score_frame.pack(pady=10)
        self.score_frame.pack_propagate(False)
        
        self.combined_lbl = ctk.CTkLabel(self.score_frame, text="COMBINED STRESS SCORE", font=ctk.CTkFont(size=12, weight="normal"))
        self.combined_lbl.pack(pady=(15,0))
        
        self.combined_score_val = ctk.CTkLabel(self.score_frame, text="0.0%", font=ctk.CTkFont(size=48, weight="bold"), text_color="#3a7ebf")
        self.combined_score_val.pack()

        # --- BREAKDOWN TRACKERS ---
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_frame.pack(pady=20, fill="x", padx=40)
        self.grid_frame.columnconfigure((0, 1, 2), weight=1)

        # Card 1: Facial 
        self.face_card = ctk.CTkFrame(self.grid_frame, height=180)
        self.face_card.grid(row=0, column=0, padx=10, sticky="nsew")
        ctk.CTkLabel(self.face_card, text="👁️ FACIAL STRESS", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.face_val_lbl = ctk.CTkLabel(self.face_card, text="0%", font=ctk.CTkFont(size=28, weight="bold"))
        self.face_val_lbl.pack(pady=10)
        self.face_lbl_desc = ctk.CTkLabel(self.face_card, text="Waiting for OpenFace...", font=ctk.CTkFont(size=11), wraplength=150)
        self.face_lbl_desc.pack()

        # Card 2: Voice
        self.voice_card = ctk.CTkFrame(self.grid_frame, height=180)
        self.voice_card.grid(row=0, column=1, padx=10, sticky="nsew")
        ctk.CTkLabel(self.voice_card, text="🎙️ VOICE STRESS", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.voice_val_lbl = ctk.CTkLabel(self.voice_card, text="0%", font=ctk.CTkFont(size=28, weight="bold"))
        self.voice_val_lbl.pack(pady=10)
        self.voice_lbl_desc = ctk.CTkLabel(self.voice_card, text="Listening to mic...", font=ctk.CTkFont(size=11), wraplength=150)
        self.voice_lbl_desc.pack()

        # Card 3: Behavior
        self.behavior_card = ctk.CTkFrame(self.grid_frame, height=180)
        self.behavior_card.grid(row=0, column=2, padx=10, sticky="nsew")
        ctk.CTkLabel(self.behavior_card, text="⌨️ BEHAVIORAL", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.behavior_val_lbl = ctk.CTkLabel(self.behavior_card, text="0%", font=ctk.CTkFont(size=28, weight="bold"))
        self.behavior_val_lbl.pack(pady=10)
        self.behavior_lbl_desc = ctk.CTkLabel(self.behavior_card, text="Keys/Clicks active", font=ctk.CTkFont(size=11), wraplength=150)
        self.behavior_lbl_desc.pack()

        # --- STATUS BAR ---
        self.status_lbl = ctk.CTkLabel(self, text="System status: Booting pipeline...", font=ctk.CTkFont(size=12), text_color="gray")
        self.status_lbl.pack(side="bottom", pady=15)

        # Start the background data UI loops safely
        self.update_ui_loop()

    def update_ui_loop(self):
        """Pulls the background snapshot values safely from assigned widget references."""
        try:
            snapshot = STATE.snapshot()
            
            behavior = BehaviorMonitor.score(snapshot)
            voice = VoiceMonitor.score(snapshot)
            facial = FacialMonitor.score(snapshot)

            # 1. Update Behavior UI Card safely
            self.behavior_val_lbl.configure(text=f"{behavior.get('score', 0)}%")
            self.behavior_lbl_desc.configure(
                text=f"Typing: {behavior.get('typing_speed', 0)}/s\nClicks: {behavior.get('clicks', 0)}\nDel: {behavior.get('backspaces', 0)}"
            )

            # 2. Update Voice UI Card safely
            v_score = voice.get("score")
            if v_score is not None:
                self.voice_val_lbl.configure(text=f"{int(v_score)}%")
                if voice.get("top_emotions"):
                    self.voice_lbl_desc.configure(text=f"Primary: {voice['top_emotions']}")
            
            # 3. Reading the live OpenFace output telemetry safely from file fallback rows
            csv_path = r"C:\hackathon\openface\openface_output.csv"
            if os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # Grab the absolute last line of telemetry data generated by OpenFace
                        last_line = lines[-1].strip().split(",")
                        headers = lines[0].strip().split(",")
                        if len(last_line) == len(headers):
                            frame_data = dict(zip(headers, last_line))
                            
                            # Calculate an live face score based on tracking action units (AU12=Smile, AU04=Frown)
                            try:
                                smile = float(frame_data.get(" AU12_r", 0.0))
                                frown = float(frame_data.get(" AU04_r", 0.0))
                                calculated_facial_stress = min(100, max(0, int((frown * 35) - (smile * 20) + 15)))
                                
                                self.face_val_lbl.configure(text=f"{calculated_facial_stress}%")
                                self.face_lbl_desc.configure(text="Tracking expressions live")
                                facial["score"] = calculated_facial_stress
                            except Exception:
                                pass

            # 4. Calculate and Update Combined Dashboard Metric
            available_scores = [s for s in (behavior.get("score"), voice.get("score"), facial.get("score")) if s is not None]
            if available_scores:
                combined = round(sum(available_scores) / len(available_scores), 1)
                self.combined_score_val.configure(text=f"{combined}%")
                
                # Dynamically change card alert highlights based on total stress
                if combined < 30:
                    self.combined_score_val.configure(text_color="#2ecc71") # Green (Calm)
                    self.status_lbl.configure(text="Status: User is completely calm.")
                elif combined < 60:
                    self.combined_score_val.configure(text_color="#f1c40f") # Yellow (Moderate)
                    self.status_lbl.configure(text="Status: Moderate tension detected.")
                else:
                    self.combined_score_val.configure(text_color="#e74c3c") # Red (High Stress Alert)
                    self.status_lbl.configure(text="Status: High stress baseline reached!")
        except Exception:
            pass

        # Keep UI frame looping infinitely every 500ms
        self.after(500, self.update_ui_loop)

def start_asyncio_loop():
    """Runs your teammate's async engine loop in a clean background thread."""
    asyncio.run(main())

if __name__ == "__main__":
    # Start teammate's background tracking loop (Mouse, Keyboard, Hume API)
    threading.Thread(target=start_asyncio_loop, daemon=True).start()
    
    # Launch your custom visual UI dashboard
    app = TiltDashboardUI()
    app.mainloop()
