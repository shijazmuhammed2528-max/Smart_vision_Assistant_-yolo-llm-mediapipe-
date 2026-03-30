
    # ── IMPORTS ───────────────────────────────────────────────────────────────────
# Standard library + OpenCV + MediaPipe + PyTorch + YOLO + Groq LLM client
import cv2, mediapipe as mp, numpy as np, threading, base64, asyncio, tempfile, os, tkinter as tk
from tkinter import simpledialog
import torch
from ultralytics import YOLO
from groq import Groq

# ── DEVICE SETUP ──────────────────────────────────────────────────────────────
# Automatically use GPU (CUDA) if available, otherwise fall back to CPU.
# YOLO and torch operations will run on this device.
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"[Device] {DEVICE.upper()}  {GPU_NAME}")

# ── OPTIONAL MIC ──────────────────────────────────────────────────────────────
# speech_recognition is optional — if not installed, mic input is disabled
# and the app falls back to a Tkinter text dialog for questions.
try: import speech_recognition as sr; MIC_AVAILABLE = True
except ImportError: MIC_AVAILABLE = False

# ── TTS ENGINE SELECTION ──────────────────────────────────────────────────────
# Tries three TTS engines in priority order and picks the first available:
#   1. edge-tts  — Microsoft neural voices (online, highest quality)
#   2. gTTS      — Google Text-to-Speech (online, good quality)
#   3. pyttsx3   — Offline system voices (no internet needed, basic quality)
# Both edge-tts and gTTS need pygame to play the generated MP3 audio.
TTS_ENGINE = TTS_AVAILABLE = None
for _eng in ("edge", "gtts", "pyttsx3"):
    try:
        if _eng in ("edge", "gtts"): import pygame; pygame.mixer.init()
        if _eng == "edge": import edge_tts
        elif _eng == "gtts": from gtts import gTTS
        else: import pyttsx3
        TTS_ENGINE, TTS_AVAILABLE = _eng, True
        print(f"[TTS] {_eng} — ACTIVE"); break
    except ImportError: pass
if not TTS_AVAILABLE: print("[WARN] No TTS. Install: pip install edge-tts pygame")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# API key and model for the Groq LLM (vision-capable multimodal model).
# YOLO_EVERY controls how often YOLO runs: every frame on GPU, every 4 on CPU.
# SKIP_LABELS filters out detections we don't want users to select (e.g. person).
GROQ_API_KEY   = "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"
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"  # AI model vision and responce.
YOLO_MODEL, SKIP_LABELS, YOLO_CONF = "yolo26l.pt", {"person"}, 0.40
EDGE_TTS_VOICE = "en-IN-NeerjaNeural"
YOLO_EVERY     = 1 if DEVICE == "cuda" else 4
CAM_W, CAM_H   = 1280, 720
F = cv2.FONT_HERSHEY_SIMPLEX          # shorthand font constant used across all putText calls

# ── MODEL INITIALISATION ──────────────────────────────────────────────────────
# Load and move YOLO to the selected device.
# MediaPipe Hands tracker is configured for single-hand real-time tracking.
groq_client  = Groq(api_key=GROQ_API_KEY)
yolo         = YOLO(YOLO_MODEL); yolo.to(DEVICE)
mp_hands     = mp.solutions.hands
mp_draw      = mp.solutions.drawing_utils
hand_tracker = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ── SHARED STATE ──────────────────────────────────────────────────────────────
# These globals are updated by the main loop and background threads.
# selected_* holds the currently gesture-selected object and its image crop.
# llm_* holds the AI response text and status flags.
selected_label = selected_box = selected_crop = None
llm_answer = current_question = mic_status = ""
llm_thinking = False

# ── UTILITY: IMAGE → BASE64 ───────────────────────────────────────────────────
# Encodes a cropped OpenCV image as a JPEG base64 data URL.
# This format is required by the Groq vision API for inline image input.
def crop_to_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

# ── FINGER COUNTER ────────────────────────────────────────────────────────────
# Counts how many fingers are raised using MediaPipe hand landmark positions.
# Each landmark has (x, y) normalized coordinates (0–1).
#
# Landmark index reference (MediaPipe 21-point hand model):
#   Thumb  : tip=4,  pip=3   → raised if tip.x < pip.x  (horizontal check, mirrored cam)
#   Index  : tip=8,  pip=6   → raised if tip.y < pip.y  (vertical check)
#   Middle : tip=12, pip=10  → raised if tip.y < pip.y
#   Ring   : tip=16, pip=14  → raised if tip.y < pip.y
#   Pinky  : tip=20, pip=18  → raised if tip.y < pip.y
#
# Used gestures in this app:
#   2 fingers (index + middle) → select the object under the fingertip
#   5 fingers (open palm)      → reset / deselect
def count_fingers(lm, _):
    lmList = [[i, p.x, p.y] for i, p in enumerate(lm.landmark)]

    thumb_up  = lmList[4][1]  < lmList[3][1]   # thumb tip X < thumb IP X (leftward = up, mirrored)
    index_up  = lmList[8][2]  < lmList[6][2]   # index tip Y < index PIP Y (higher on screen = up)
    middle_up = lmList[12][2] < lmList[10][2]  # middle tip Y < middle PIP Y
    ring_up   = lmList[16][2] < lmList[14][2]  # ring tip Y < ring PIP Y
    pinky_up  = lmList[20][2] < lmList[18][2]  # pinky tip Y < pinky PIP Y

    return sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

# ── HIT TEST ──────────────────────────────────────────────────────────────────
# Checks if the fingertip pixel (px, py) falls inside any YOLO bounding box.
# Returns the label and box coordinates of the first matching detection.
def hit_test(px, py, dets):
    for label, x1, y1, x2, y2, _ in dets:
        if x1 < px < x2 and y1 < py < y2: return label, (x1, y1, x2, y2)
    return None, None

# ── TTS: edge-tts ASYNC HELPER ────────────────────────────────────────────────
# Generates speech via Microsoft edge-tts, saves to a temp MP3, plays via pygame.
# Runs inside asyncio.run() from the sync _speak() wrapper below.
async def _edge_speak(text):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False); tmp.close()
    try:
        await edge_tts.Communicate(text, EDGE_TTS_VOICE).save(tmp.name)
        pygame.mixer.music.load(tmp.name); pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): await asyncio.sleep(0.05)
    finally:
        try: pygame.mixer.music.unload(); os.remove(tmp.name)
        except: pass

# ── TTS: UNIFIED SPEAK FUNCTION ───────────────────────────────────────────────
# Routes speech output to whichever TTS engine was detected at startup.
# Called from a daemon thread so it never blocks the main camera loop.
def _speak(text):
    if not TTS_AVAILABLE: return
    try:
        if TTS_ENGINE == "edge":
            asyncio.run(_edge_speak(text))
        elif TTS_ENGINE == "gtts":
            import time
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False); tmp.close()
            gTTS(text=text, lang="en", tld="co.in").save(tmp.name)
            pygame.mixer.music.load(tmp.name); pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.05)
            try: pygame.mixer.music.unload(); os.remove(tmp.name)
            except: pass
        else:
            # pyttsx3: prefer a female voice if available, then speak at 155 wpm
            e = pyttsx3.init()
            for v in e.getProperty("voices"):
                if any(n in v.name.lower() for n in ("zira","hazel","susan","helena")):
                    e.setProperty("voice", v.id); break
            e.setProperty("rate", 155); e.say(text); e.runAndWait()
    except Exception as ex: print(f"[TTS ERR] {ex}")

# ── VISION LLM QUERY ──────────────────────────────────────────────────────────
# Sends the cropped object image + user's question to the Groq vision model.
# Runs in a background thread (called from listen_and_ask or V key).
# Updates global llm_answer which draw_hud() reads and displays on screen.
def ask_llm_vision(crop, obj_label, question):
    global llm_answer, llm_thinking, current_question
    llm_thinking = True; llm_answer = "Thinking…"; current_question = question
    try:
        resp = groq_client.chat.completions.create(
            model=VISION_MODEL, temperature=1, max_completion_tokens=200, top_p=1,
            messages=[
                {"role": "system", "content": f"You are a smart camera assistant. YOLO labelled this '{obj_label}'. Answer in 1-3 sentences."},
                {"role": "user",   "content": [{"type": "text", "text": question},
                                               {"type": "image_url", "image_url": {"url": crop_to_b64(crop)}}]},
            ])
        llm_answer = resp.choices[0].message.content.strip()
        print(f"\n[AI] {llm_answer}\n")
        # Speak the answer in a separate thread so TTS doesn't block the LLM thread
        if TTS_AVAILABLE: threading.Thread(target=_speak, args=(llm_answer,), daemon=True).start()
    except Exception as ex: llm_answer = f"Error: {ex}"; print(f"[LLM ERR] {ex}")
    finally: llm_thinking = False

# ── MIC / TEXT INPUT ──────────────────────────────────────────────────────────
# Triggered by pressing Q. Tries to capture voice input first via the microphone.
# If speech_recognition is unavailable or no speech is detected,
# falls back to a Tkinter popup dialog for typed input.
# Once a question is captured, fires ask_llm_vision() in a background thread.
def listen_and_ask():
    global mic_status
    if not selected_label or selected_crop is None or llm_thinking: return
    question = None
    if MIC_AVAILABLE:
        rec = sr.Recognizer()
        try:
            with sr.Microphone() as src:
                mic_status = "Calibrating…"; rec.energy_threshold = 300
                rec.dynamic_energy_threshold = False
                rec.adjust_for_ambient_noise(src, duration=0.5)  # suppress background noise
                mic_status = "Listening…  (speak now)"
                audio = rec.listen(src, timeout=10, phrase_time_limit=10)
            mic_status = "Transcribing…"
            question = rec.recognize_google(audio, language="en-IN")
            mic_status = f'Heard: "{question}"'
        except (sr.WaitTimeoutError, sr.UnknownValueError): mic_status = "No speech — opening text box…"
        except Exception as ex: mic_status = "Mic error — opening text box…"; print(f"[MIC ERR] {ex}")
    if not question:
        # Fallback: open a simple Tkinter dialog box for typed question
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        question = simpledialog.askstring(f"Ask about: {selected_label}",
                                          f"What do you want to know about the {selected_label}?", parent=root)
        root.destroy(); mic_status = ""
    if question and question.strip():
        threading.Thread(target=ask_llm_vision, args=(selected_crop.copy(), selected_label, question.strip()), daemon=True).start()
    else: mic_status = ""

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
# Named BGR colors used consistently across all drawing functions.
C = dict(grey=(160,160,160), green=(50,220,140), amber=(80,220,255),
         purple=(120,160,255), hint=(110,110,110), mic=(80,200,255),
         q=(180,190,100), ans=(210,210,210))

# ── DRAW: YOLO BOUNDING BOXES ─────────────────────────────────────────────────
# Draws a box and label badge for each detected object.
# The currently selected object gets a thicker green border; others are grey.
def draw_detections(frame, dets, sel_box):
    for label, x1, y1, x2, y2, conf in dets:
        col = C["green"] if sel_box == (x1,y1,x2,y2) else C["grey"]
        th  = 3 if sel_box == (x1,y1,x2,y2) else 1
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, th)
        badge = f"{label}  {conf:.0%}"
        (tw, bh), _ = cv2.getTextSize(badge, F, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-bh-10), (x1+tw+10, y1), col, -1)
        cv2.putText(frame, badge, (x1+5, y1-5), F, 0.55, (0,0,0), 1, cv2.LINE_AA)

# ── DRAW: HUD OVERLAY ─────────────────────────────────────────────────────────
# Draws the semi-transparent bottom panel showing:
#   • GPU/CPU label + FPS counter (top-right corner)
#   • Active TTS engine name
#   • Key shortcut hint bar
#   • Selected object name, mic status, question text, AI answer (word-wrapped)
#   • Thumbnail of the image crop that was sent to the AI
def draw_hud(frame, h, w, fps):
    y0 = h - 220  # top edge of the bottom HUD panel

    # Semi-transparent dark background for the HUD panel
    ov = frame.copy(); cv2.rectangle(ov, (0,y0), (w,h), (15,15,15), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0,y0), (w,y0), (70,70,70), 1)  # separator line

    # GPU / FPS badge (top-right)
    gpu_col = C["green"] if DEVICE=="cuda" else C["amber"]
    lbl = f"{'GPU' if DEVICE=='cuda' else 'CPU'}  {fps:.0f} FPS"
    (gw,_),_ = cv2.getTextSize(lbl, F, 0.50, 1)
    cv2.rectangle(frame, (w-gw-20,6), (w-4,28), (20,20,20), -1)
    cv2.putText(frame, lbl, (w-gw-12,23), F, 0.50, gpu_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TTS: {TTS_ENGINE or 'OFF'}", (w-gw-12,44), F, 0.35, C["amber"], 1, cv2.LINE_AA)

    # Key shortcut hint at top of frame
    cv2.putText(frame, "1=move | 2=select | 5=reset | Q=speak | V=auto | ESC=quit", (10,22), F, 0.36, C["hint"], 1, cv2.LINE_AA)

    if selected_label:
        # Show selected object name and mic/question/answer status
        cv2.putText(frame, f"Selected: {selected_label.upper()}", (16,y0+28), F, 0.70, C["green"], 2, cv2.LINE_AA)
        if mic_status: cv2.putText(frame, mic_status, (16,y0+52), F, 0.44, C["mic"], 1, cv2.LINE_AA)
        if current_question:
            q = (current_question[:95]+"…") if len(current_question)>95 else current_question
            cv2.putText(frame, f"Q: {q}", (16,y0+74), F, 0.44, C["q"], 1, cv2.LINE_AA)

        # Word-wrap the LLM answer to fit within 80 characters per line
        words, lines, cur = llm_answer.split(), [], ""
        for wrd in words:
            cur = (cur+" "+wrd).strip() if len(cur)+len(wrd)+1 <= 80 else (lines.append(cur) or wrd)
        if cur: lines.append(cur)
        for i, ln in enumerate(lines[:5]):
            cv2.putText(frame, ln, (16,y0+100+i*24), F, 0.44, C["ans"], 1, cv2.LINE_AA)

        # Thumbnail of the cropped image that was sent to the AI (bottom-right of HUD)
        if selected_crop is not None and selected_crop.size > 0:
            thumb = cv2.resize(selected_crop, (90,68))
            tx, ty = w-106, y0+144
            frame[ty:ty+68, tx:tx+90] = thumb
            cv2.rectangle(frame, (tx-1,ty-1), (tx+91,ty+69), C["green"], 1)
            cv2.putText(frame, "sent to AI", (tx,ty-4), F, 0.30, C["green"], 1, cv2.LINE_AA)
    else:
        # No selection yet — show usage hint
        cv2.putText(frame, "Point at object → raise 2 fingers to select", (16,y0+110), F, 0.58, (120,120,120), 1, cv2.LINE_AA)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
# Opens the webcam and runs the per-frame pipeline:
#   1. Read + flip frame
#   2. Run YOLO object detection (throttled by YOLO_EVERY)
#   3. Run MediaPipe hand tracking
#   4. Map finger count to gesture actions (select / reset)
#   5. Draw detections + HUD overlay
#   6. Handle keyboard shortcuts (Q=speak, V=auto-ask, ESC=quit)
def main():
    global selected_label, selected_box, selected_crop, llm_answer, current_question, mic_status

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened(): print("ERROR: Cannot open webcam."); return

    frame_count, detections, prev_tick, fps = 0, [], cv2.getTickCount(), 0.0
    print(f"\n[Ready] Device={DEVICE.upper()}  TTS={TTS_ENGINE}  Mic={'ON' if MIC_AVAILABLE else 'OFF'}")
    print("Q=speak  V=auto-ask  ESC=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: print("ERROR: webcam read failed."); break
        frame = cv2.flip(frame, 1); h, w = frame.shape[:2]  # mirror for natural interaction

        # FPS calculation using OpenCV's high-res tick counter
        tick = cv2.getTickCount(); fps = cv2.getTickFrequency() / (tick - prev_tick); prev_tick = tick

        # YOLO detection — throttled to save CPU/GPU on slower hardware
        if frame_count % YOLO_EVERY == 0:
            results    = yolo(frame, verbose=False, conf=YOLO_CONF, device=DEVICE)[0]
            detections = [(yolo.names[int(b.cls[0])], *map(int,b.xyxy[0]), float(b.conf[0]))
                          for b in results.boxes if yolo.names[int(b.cls[0])] not in SKIP_LABELS]
        frame_count += 1

        # MediaPipe hand tracking — runs every frame for smooth gesture response
        mp_result = hand_tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if mp_result.multi_hand_landmarks:
            hand_lm = mp_result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec((70,70,70),1,2), mp_draw.DrawingSpec((70,70,70),1))

            fingers = count_fingers(hand_lm, None)
            # Landmark 8 = index fingertip — used as the pointing/selection cursor
            px, py  = int(hand_lm.landmark[8].x*w), int(hand_lm.landmark[8].y*h)

            if fingers == 2:
                # 2 fingers: point at an object to select it and save its crop for AI
                label, box = hit_test(px, py, detections)
                if label and label != selected_label:
                    x1,y1,x2,y2 = box; crop = frame[max(0,y1):y2, max(0,x1):x2].copy()
                    if crop.size > 0:
                        selected_label, selected_box, selected_crop = label, box, crop
                        llm_answer = current_question = mic_status = ""
                        print(f"[SELECT] {label}")
            elif fingers == 5:
                # 5 fingers (open palm): reset all selection and AI state
                if selected_label: print("[RESET]")
                selected_label = selected_box = selected_crop = None
                llm_answer = current_question = mic_status = ""

            # Fingertip cursor: color reflects current gesture
            col = {1:C["amber"],2:C["green"],5:C["purple"]}.get(fingers,C["grey"])
            cv2.circle(frame, (px,py), 16, col, 2); cv2.circle(frame, (px,py), 5, col, -1)

        draw_detections(frame, detections, selected_box)
        draw_hud(frame, h, w, fps)
        cv2.imshow("Smart Vision Assistant", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: break  # ESC → quit
        elif key in (ord('q'),ord('Q')) and selected_label and not llm_thinking:
            # Q → capture voice/text question and send to LLM
            threading.Thread(target=listen_and_ask, daemon=True).start()
        elif key in (ord('v'),ord('V')) and selected_label and selected_crop is not None and not llm_thinking:
            # V → auto-ask a default description question about the selected object
            q = f"What exactly is this {selected_label} and what is it typically used for?"
            threading.Thread(target=ask_llm_vision, args=(selected_crop.copy(),selected_label,q), daemon=True).start()

    # Cleanup: release camera, close windows, shut down TTS mixer
    cap.release(); cv2.destroyAllWindows(); hand_tracker.close()
    if TTS_ENGINE in ("edge","gtts"): pygame.mixer.quit()
    print("Goodbye.")

if __name__ == "__main__":
    main()
