

# ── standard library ──────────────────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import tkinter as tk
from tkinter import simpledialog

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    DEVICE     = "cuda"
    GPU_NAME   = torch.cuda.get_device_name(0)
    VRAM_GB    = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] CUDA enabled  →  {GPU_NAME}  ({VRAM_GB:.1f} GB VRAM)")
else:
    DEVICE   = "cpu"
    GPU_NAME = "N/A"
    print("[GPU] CUDA NOT available — falling back to CPU.")
    print("      Check: python -c \"import torch; print(torch.cuda.is_available())\"")

# ── YOLO (GPU-accelerated) ────────────────────────────────────────────────────
from ultralytics import YOLO

# ── Groq vision client ────────────────────────────────────────────────────────
# NOTE: if you have a file called  groq.py  in THIS folder it will shadow the
# real groq package → rename it:   ren groq.py groq_old.py
from groq import Groq

# ── optional: mic STT ─────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    print("[WARN] speech_recognition not installed — mic disabled.")

# ── optional: TTS ─────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed — spoken answers disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = "API_KEY"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
YOLO_MODEL   = "yolo26l.pt"       # auto-downloads on first run
SKIP_LABELS  = {"person"}         # never detect / select people
YOLO_CONF    = 0.40               # detection confidence threshold

# ── GPU vs CPU frame interval ─────────────────────────────────────────────────
# On GPU we can afford to run YOLO every single frame.
# On CPU we throttle to every 4 frames to stay smooth.
YOLO_EVERY   = 1 if DEVICE == "cuda" else 4

CAM_W, CAM_H = 1280, 720


# ─────────────────────────────────────────────────────────────────────────────
# INIT MODELS
# ─────────────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)

# Pass device="cuda" → YOLO loads weights onto GPU automatically
yolo = YOLO(YOLO_MODEL)
yolo.to(DEVICE)                    # ← moves model to GPU
print(f"[YOLO] Running on {DEVICE.upper()}")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hand_tracker = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────

selected_label   = None
selected_box     = None
selected_crop    = None   # BGR numpy array sent to vision model
llm_answer       = ""
llm_thinking     = False
current_question = ""
pointer_pos      = (0, 0)
mic_status       = ""
fps_display      = 0.0    # live FPS shown on HUD


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: BGR crop → base-64 JPEG data-URI
# ─────────────────────────────────────────────────────────────────────────────

def crop_to_b64(crop_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# GESTURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_fingers(hand_landmarks, handedness) -> int:
    lm       = hand_landmarks.landmark
    tips     = [4,  8, 12, 16, 20]
    pips     = [3,  6, 10, 14, 18]
    count    = 0
    is_right = handedness.classification[0].label == "Right"
    if is_right:
        if lm[tips[0]].x < lm[pips[0]].x: count += 1
    else:
        if lm[tips[0]].x > lm[pips[0]].x: count += 1
    for i in range(1, 5):
        if lm[tips[i]].y < lm[pips[i]].y: count += 1
    return count


def get_pointer(hand_landmarks, fw: int, fh: int):
    t = hand_landmarks.landmark[8]
    return int(t.x * fw), int(t.y * fh)


def hit_test(px: int, py: int, detections: list):
    for label, x1, y1, x2, y2, conf in detections:
        if x1 < px < x2 and y1 < py < y2:
            return label, (x1, y1, x2, y2)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# VISION LLM
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm_vision(crop_bgr: np.ndarray, object_label: str, question: str):
    """Send cropped image + question to Llama-4 Scout. Runs in a thread."""
    global llm_answer, llm_thinking, current_question

    llm_thinking     = True
    llm_answer       = "Thinking…"
    current_question = question

    try:
        image_url = crop_to_b64(crop_bgr)
        resp = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a smart camera assistant. "
                        f"YOLO labelled this object as '{object_label}'. "
                        "Look at the image and answer the user's question "
                        "in 1-3 concise sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=1,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
        )
        answer     = resp.choices[0].message.content.strip()
        llm_answer = answer
        print(f"\n[AI] {answer}\n")
        if TTS_AVAILABLE:
            threading.Thread(target=_speak, args=(answer,), daemon=True).start()

    except Exception as exc:
        llm_answer = f"Error: {exc}"
        print(f"[LLM ERROR] {exc}")
    finally:
        llm_thinking = False


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────

def _speak(text: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:
        print(f"[TTS ERROR] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MICROPHONE INPUT
# ─────────────────────────────────────────────────────────────────────────────

def listen_and_ask():
    """Record spoken question → STT → vision LLM. Falls back to text box."""
    global mic_status

    if not selected_label or selected_crop is None or llm_thinking:
        return

    question = None

    if MIC_AVAILABLE:
        rec = sr.Recognizer()
        try:
            with sr.Microphone() as src:
                mic_status = "Calibrating mic…  (1 sec)"
                print("[MIC] Calibrating…")
                rec.energy_threshold       = 300   # fixed low threshold — more sensitive
                rec.dynamic_energy_threshold = False  # don't let it drift upward
                rec.adjust_for_ambient_noise(src, duration=0.5)
                print(f"[MIC] Energy threshold set to: {rec.energy_threshold:.0f}")
                mic_status = "Listening…  (speak now)"
                print("[MIC] Listening…")
                audio = rec.listen(src, timeout=10, phrase_time_limit=10)
            mic_status = "Transcribing…"
            question   = rec.recognize_google(audio, language="en-IN")
            mic_status = f'Heard: "{question}"'
            print(f"[MIC] Heard: {question}")
        except sr.WaitTimeoutError:
            mic_status = "No speech detected — opening text box…"
            print("[MIC] WaitTimeoutError: no speech heard within timeout")
        except sr.UnknownValueError:
            mic_status = "Didn't catch that — opening text box…"
            print("[MIC] UnknownValueError: speech heard but not understood")
        except Exception as exc:
            mic_status = "Mic error — opening text box…"
            print(f"[MIC ERROR] {exc}")

    if not question:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        question = simpledialog.askstring(
            title=f"Ask about: {selected_label}",
            prompt=f"What do you want to know about the {selected_label}?",
            parent=root,
        )
        root.destroy()
        mic_status = ""

    if question and question.strip():
        snap = selected_crop.copy()
        threading.Thread(
            target=ask_llm_vision,
            args=(snap, selected_label, question.strip()),
            daemon=True,
        ).start()
    else:
        mic_status = ""


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────────────────────────────────────

C_GREY   = (160, 160, 160)
C_GREEN  = ( 50, 220, 140)
C_AMBER  = ( 80, 220, 255)
C_PURPLE = (120, 160, 255)
C_HINT   = (110, 110, 110)
C_MIC    = ( 80, 200, 255)
C_Q      = (180, 190, 100)
C_ANS    = (210, 210, 210)
C_GPU    = ( 50, 220, 140)   # green when on GPU
C_CPU    = ( 80, 200, 255)   # amber when on CPU


def draw_detections(frame, detections, sel_box):
    for label, x1, y1, x2, y2, conf in detections:
        is_sel = sel_box == (x1, y1, x2, y2)
        color  = C_GREEN if is_sel else C_GREY
        thick  = 3 if is_sel else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        badge = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, badge, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_cursor(frame, pos, fingers):
    cmap  = {1: C_AMBER, 2: C_GREEN, 5: C_PURPLE}
    color = cmap.get(fingers, C_GREY)
    cv2.circle(frame, pos, 16, color, 2)
    cv2.circle(frame, pos,  5, color, -1)


def wrap_text(text: str, max_chars: int = 90) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines


def draw_hud(frame, h, w, fps):
    panel_h = 220          # taller panel to fit full answer
    y0      = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), (70, 70, 70), 1)

    # ── GPU / FPS badge (top-right) ───────────────────────────────────────────
    gpu_color = C_GPU if DEVICE == "cuda" else C_CPU
    gpu_label = f"{'GPU' if DEVICE == 'cuda' else 'CPU'}  {fps:.0f} FPS"
    (gw, gh), _ = cv2.getTextSize(gpu_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (w - gw - 20, 6), (w - 4, 28), (20, 20, 20), -1)
    cv2.putText(frame, gpu_label, (w - gw - 12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, gpu_color, 1, cv2.LINE_AA)

    # ── Bottom panel content ──────────────────────────────────────────────────
    if selected_label:
        # Row 1: selected label
        cv2.putText(frame, f"Selected:  {selected_label.upper()}",
                    (16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, C_GREEN, 2, cv2.LINE_AA)

        # Row 2: mic status (only when active)
        if mic_status:
            cv2.putText(frame, mic_status,
                        (16, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, C_MIC, 1, cv2.LINE_AA)

        # Row 3: question (truncated to one line)
        if current_question:
            q_display = (current_question[:95] + "…") if len(current_question) > 95 else current_question
            cv2.putText(frame, f"Q: {q_display}",
                        (16, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, C_Q, 1, cv2.LINE_AA)

        # Rows 4-8: AI answer — up to 5 lines, 80 chars each
        answer_lines = wrap_text(llm_answer, max_chars=80)[:5]
        for i, line in enumerate(answer_lines):
            cv2.putText(frame, line,
                        (16, y0 + 100 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_ANS, 1, cv2.LINE_AA)

        # Object crop thumbnail — bottom-right corner of the panel
        if selected_crop is not None and selected_crop.size > 0:
            thumb      = cv2.resize(selected_crop, (90, 68))
            tx, ty     = w - 106, y0 + panel_h - 76
            frame[ty:ty+68, tx:tx+90] = thumb
            cv2.rectangle(frame, (tx-1, ty-1), (tx+91, ty+69), C_GREEN, 1)
            cv2.putText(frame, "sent to AI", (tx, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame,
                    "Point at an object  →  raise 2 fingers to select it",
                    (16, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (120, 120, 120), 1, cv2.LINE_AA)

    # Top hint bar
    cv2.putText(frame,
                "1 finger: move  |  2 fingers: select  |  5 fingers: reset  "
                "|  Q: speak question  |  V: auto-ask  |  ESC: quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_HINT, 1, cv2.LINE_AA)



# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global selected_label, selected_box, selected_crop
    global llm_answer, current_question, pointer_pos, mic_status, fps_display

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try VideoCapture(1).")
        return

    frame_count  = 0
    detections   = []
    prev_tick    = cv2.getTickCount()
    fps          = 0.0

    print("\n" + "═" * 60)
    print("  Smart Vision Assistant  —  GPU Edition")
    print("═" * 60)
    print(f"  Device       : {DEVICE.upper()}  {('— ' + GPU_NAME) if DEVICE == 'cuda' else ''}")
    print(f"  YOLO interval: every {YOLO_EVERY} frame(s)")
    print(f"  Vision model : {VISION_MODEL}")
    print(f"  Mic / STT    : {'ON' if MIC_AVAILABLE else 'OFF'}")
    print(f"  TTS output   : {'ON' if TTS_AVAILABLE  else 'OFF'}")
    print("─" * 60)
    print("  Q → speak question   V → auto-ask   ESC → quit")
    print("═" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── FPS calculation ───────────────────────────────────────────────────
        tick      = cv2.getTickCount()
        fps       = cv2.getTickFrequency() / (tick - prev_tick)
        prev_tick = tick

        # ── YOLO on GPU ───────────────────────────────────────────────────────
        if frame_count % YOLO_EVERY == 0:
            # device=DEVICE tells ultralytics to run inference on GPU
            results    = yolo(frame, verbose=False, conf=YOLO_CONF, device=DEVICE)[0]
            detections = []
            for box in results.boxes:
                label = yolo.names[int(box.cls[0])]
                if label in SKIP_LABELS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((label, x1, y1, x2, y2, conf))

        frame_count += 1

        # ── MediaPipe hand tracking ───────────────────────────────────────────
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = hand_tracker.process(rgb)
        fingers   = 0

        if mp_result.multi_hand_landmarks:
            hand_lm    = mp_result.multi_hand_landmarks[0]
            handedness = mp_result.multi_handedness[0]

            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(70, 70, 70), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(70, 70, 70), thickness=1),
            )

            fingers     = count_fingers(hand_lm, handedness)
            pointer_pos = get_pointer(hand_lm, w, h)

            if fingers == 1:
                pass   # move mode

            elif fingers == 2:
                label, box = hit_test(pointer_pos[0], pointer_pos[1], detections)
                if label and label != selected_label:
                    x1, y1, x2, y2 = box
                    crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size > 0:
                        selected_label   = label
                        selected_box     = box
                        selected_crop    = crop
                        llm_answer       = ""
                        current_question = ""
                        mic_status       = ""
                        print(f"[SELECT] {label}")

            elif fingers == 5:
                if selected_label:
                    print("[RESET] Cleared.")
                selected_label   = None
                selected_box     = None
                selected_crop    = None
                llm_answer       = ""
                current_question = ""
                mic_status       = ""

            draw_cursor(frame, pointer_pos, fingers)

        # ── Render ────────────────────────────────────────────────────────────
        draw_detections(frame, detections, selected_box)
        draw_hud(frame, h, w, fps)
        cv2.imshow("Smart Vision Assistant  [GPU]", frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Exiting.")
            break

        elif key in (ord('q'), ord('Q')):
            if selected_label and not llm_thinking:
                threading.Thread(target=listen_and_ask, daemon=True).start()
            elif not selected_label:
                print("[INFO] Select an object first (2 fingers).")

        elif key in (ord('v'), ord('V')):
            if selected_label and selected_crop is not None and not llm_thinking:
                auto_q = (
                    f"What exactly is this {selected_label} in the image "
                    f"and what is it typically used for?"
                )
                snap = selected_crop.copy()
                threading.Thread(
                    target=ask_llm_vision,
                    args=(snap, selected_label, auto_q),
                    daemon=True,
                ).start()
            elif not selected_label:
                print("[INFO] Select an object first (2 fingers).")

    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()

