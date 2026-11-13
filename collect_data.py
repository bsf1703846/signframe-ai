"""
SignFrame AI - Collect YOUR OWN webcam data
Why: Kaggle images are clean studio photos. Your webcam is real life.
50-100 images per letter from your own webcam fixes domain shift.

Usage:
    python collect_data.py --letter A --count 100
    python collect_data.py --letter B --count 100

Controls while running:
    SPACE = save this frame as a sample
    Q     = quit
    The script only saves when MediaPipe finds a hand.

Output:
    data/webcam/A/0001.jpg, 0002.jpg, ...
    data/webcam/A_landmarks.npy  (also saved for landmark training)

Tip: collect in 2-3 different rooms / lightings. That variety is the training data.
"""
import argparse
import os
import time

import cv2

from src.hand_utils import get_hand_detector, landmarks_to_vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--letter", required=True, help="Letter to collect, e.g. A")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default="data/webcam")
    args = p.parse_args()

    letter = args.letter.upper()
    out_dir = os.path.join(args.out, letter)
    os.makedirs(out_dir, exist_ok=True)

    # Resume from existing files
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    saved = existing
    print(f"Collecting letter '{letter}' - already have {existing}, target {args.count}")
    print("Show the sign, press SPACE to save, Q to quit.")

    cap = cv2.VideoCapture(args.camera)
    hands = get_hand_detector()
    last_save = 0

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_found = bool(result.multi_hand_landmarks)
        if hand_found:
            from src.hand_utils import mp_drawing, mp_hands
            mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"{letter}: {saved}/{args.count}  [SPACE save | Q quit]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Hand detected" if hand_found else "No hand - show your hand",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_found else (0, 0, 255), 2)
        cv2.imshow("Collect Data - SignFrame", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if key == ord(" ") and hand_found and now - last_save > 0.25:
            # Save image + landmarks
            path = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            vec = landmarks_to_vector(result.multi_hand_landmarks[0])
            import numpy as np
            np.save(os.path.join(out_dir, f"{saved:04d}.npy"), vec)
            saved += 1
            last_save = now
            print(f"Saved {path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} samples for '{letter}' in {out_dir}")


if __name__ == "__main__":
    main()
