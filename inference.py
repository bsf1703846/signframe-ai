"""
SignFrame AI - Real-time inference (THE APP)

Usage:
  python inference.py --model models/landmark_mlp.pt
  python inference.py --model models/landmark_mlp.pt --no-speech   (no voice)

How it works:
  Webcam -> MediaPipe Hands -> 63 landmarks -> MLP -> letter
  Letter is only CONFIRMED after it is stable for HOLD_FRAMES frames
  with confidence above CONF_THRESHOLD. This stops flicker.

Controls:
  SPACE = speak the current word out loud
  C     = clear text
  BACKSPACE = delete last letter
  Q     = quit

  Tip: hold a sign steady for ~1 second. Don't rush through letters.
"""
import argparse
import collections

import cv2
import numpy as np
import torch

from src.hand_utils import get_hand_detector, landmarks_to_vector, mp_drawing, mp_hands
from src.model import LETTERS_24, LandmarkMLP

CONF_THRESHOLD = 0.85
HOLD_FRAMES = 20      # ~0.7 sec at 30 fps
COOLDOWN_FRAMES = 15  # don't repeat the same letter immediately


def speak(text):
    if not text.strip():
        return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech failed (text still saved): {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/landmark_mlp.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-speech", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    letters = ckpt.get("letters", LETTERS_24)
    print(f"Loaded {args.model} - val_acc={ckpt.get('val_acc', 'n/a')}")

    model = LandmarkMLP(num_classes=len(letters)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    hands = get_hand_detector()
    history = collections.deque(maxlen=HOLD_FRAMES)
    text = ""
    last_confirmed = ""
    cooldown = 0
    no_hand_frames = 0

    print("SignFrame running. Hold a sign steady ~1 sec to confirm it. SPACE=speak, C=clear, Q=quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred_letter, conf = "-", 0.0
        if result.multi_hand_landmarks:
            no_hand_frames = 0
            lm = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            vec = landmarks_to_vector(lm)
            with torch.no_grad():
                logits = model(torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device))
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)
                pred_letter, conf = letters[int(idx)], float(conf)
            history.append(pred_letter if conf > CONF_THRESHOLD else None)
        else:
            no_hand_frames += 1
            history.append(None)
            # Auto-speak the word after 2 seconds with no hand
            if no_hand_frames == 60 and text.strip() and not args.no_speech:
                print(f"Speaking: {text}")
                speak(text)

        # Confirm: same letter for HOLD_FRAMES in a row
        if cooldown > 0:
            cooldown -= 1
        if len(history) == HOLD_FRAMES and cooldown == 0:
            vals = [h for h in history if h is not None]
            if len(vals) == HOLD_FRAMES and len(set(vals)) == 1 and vals[0] != last_confirmed:
                text += vals[0]
                last_confirmed = vals[0]
                cooldown = COOLDOWN_FRAMES
                history.clear()
                print(f"Confirmed: {vals[0]}  ->  Text: {text}")

        # UI
        cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {pred_letter}  ({conf:.2f})", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {text}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE speak | C clear | BACKSPACE delete | Q quit", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("SignFrame AI - ASL Fingerspelling", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            print(f"Speaking: {text}")
            if not args.no_speech:
                speak(text)
        elif key == ord("c"):
            text = ""; last_confirmed = ""; history.clear()
        elif key == 8:  # backspace
            text = text[:-1]; last_confirmed = ""

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final text: {text}")


if __name__ == "__main__":
    main()
