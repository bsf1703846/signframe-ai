# SignFrame AI — Real-Time ASL Fingerspelling

Real-time American Sign Language (ASL) fingerspelling recognition with Python, PyTorch, OpenCV, and MediaPipe.

**Status: Complete — trained, tested live, and working on webcam.**

A custom model recognizes **24 static ASL handshapes** through a webcam, converts confirmed predictions into text, and speaks the word out loud.

> J and Z are excluded — they need motion, they aren't static shapes.

## Results — verified 14 November 2026

| Test | Result |
|---|---|
| Dataset | Kaggle ASL Alphabet, 87,000 images (24 letters × 3,000) |
| Model | LandmarkMLP, 63 → 128 → 64 → 24, trained 30 epochs on CPU |
| Validation accuracy | **0.99 (99%)** |
| Validation samples | 8,313 |
| Macro avg / weighted avg | 0.99 / 0.99 |
| Per-letter F1 | 0.96–1.00 for most letters (lowest: M 0.96, N 0.96) |
| Live webcam test | Working — hand landmarks tracked in real time, text built letter by letter (see Demo below) |

Training was run on Windows 11, Python 3.11.9, PyTorch 2.15, MediaPipe 0.10.14. The trained model is saved as `models/landmark_mlp.pt`.

## Demo — live on webcam

The app was tested live on a laptop webcam on 14 November 2026:

- **Prediction: B (0.64) — Text: SHE** — S, H, and E were spelled and confirmed, then B was recognised live, with hand landmarks tracked on screen.
- **Prediction: I (0.40) — Text: SHEHRNH** — further letters recognised live, with lower confidence than on Kaggle data (see Domain shift below).

A 30-second demo video is pending — the live screenshots above are the verified demo evidence for now. To record your own: run the app, press `Win + Alt + R` to start Windows screen recording, spell H-E-L-L-O, press SPACE to hear it speak, then `Win + Alt + R` again to stop. The video saves to `Videos > Captures`.

Run it yourself:

```bash
python inference.py --model models/landmark_mlp.pt
# Hold a sign steady ~1 second to confirm it.
# SPACE = speak, C = clear, BACKSPACE = delete, Q = quit
```

## What you get

| File | What it does |
|---|---|
| `collect_data.py` | Capture your own webcam samples per letter (fixes domain shift) |
| `train_landmark.py` | **Start here.** Trains a tiny MLP on 21 hand landmarks (63 numbers). Fast, light, robust. |
| `train_image.py` | Step 2. Transfer-learning image CNN (MobileNetV2) on hand crops. |
| `inference.py` | The real-time app. Webcam → prediction → text → speech. |
| `src/model.py` | LandmarkMLP, SimpleCNN, MobileNetV2 + the 24-letter list |
| `src/hand_utils.py` | MediaPipe helpers: landmark extraction + wrist normalization |

## Quick start — 15 minutes to a working demo

```bash
# 1. Install
pip install -r requirements.txt

# 2. Check the code runs (no data needed)
python train_landmark.py --smoke-test

# 3. Get data — Kaggle ASL Alphabet Dataset (87,000 images)
#    Download from Kaggle, unzip so you have:
#    data/asl_alphabet_train/A/0001.jpg ...
#    data/asl_alphabet_train/B/0001.jpg ...

# 4. Train the landmark model (CPU is fine, ~30-45 min for full data)
python train_landmark.py --data data/asl_alphabet_train --epochs 30

# 5. Run it live
python inference.py --model models/landmark_mlp.pt
# Hold a sign steady ~1 second to confirm it.
# SPACE = speak, C = clear, BACKSPACE = delete, Q = quit
```

Notes from the real build: use Python 3.11.9 (not 3.15 — too new for PyTorch / MediaPipe), pin `mediapipe==0.10.14` (1.x removed the `mp.solutions` API), and keep the Kaggle folders at exactly `data/asl_alphabet_train/A/` — not double-nested.

## Domain shift — what I learned

1. A model that is 99% on clean Kaggle photos drops to 0.40–0.64 confidence on a real webcam — same hand, different light, background, and camera.
2. Test accuracy alone is not enough: a model must be tested on the webcam it will actually run on, not just on the dataset it trained on.
3. The fix is own data: collect webcam samples with `collect_data.py` and retrain with `--webcam data/webcam` mixed in, so the model learns your hand, your room, and your light.

The landmark model already helps a lot: landmarks are normalized relative to the wrist, so hand position and size don't matter, and background is ignored completely. To improve webcam accuracy further:

```bash
python collect_data.py --letter A --count 100
# Repeat for each letter, in 2-3 different rooms, then:
python train_landmark.py --data data/asl_alphabet_train --webcam data/webcam --epochs 30
```

## How the real-time logic works

```
Webcam frame
  -> MediaPipe Hands (21 landmarks)
  -> Normalize landmarks (wrist = origin, scale = 1)
  -> MLP predicts a letter + confidence
  -> Only confirm if confidence > 0.85 for 20 frames in a row (~0.7 sec)
  -> Add letter to text, cooldown so it doesn't repeat
  -> No hand for 2 sec = auto-speak the word (pyttsx3, offline)
```

That hold-to-confirm step is what stops flicker. Without it, the text jumps around every frame.

## Project structure

```
SignFrame-AI-starter/
├── README.md
├── requirements.txt
├── collect_data.py
├── train_landmark.py
├── train_image.py
├── inference.py
├── src/
│   ├── model.py
│   └── hand_utils.py
├── data/
│   ├── asl_alphabet_train/   # Kaggle data goes here (not included)
│   └── webcam/               # Your own samples go here
└── models/
    └── landmark_mlp.pt       # Trained model — 99% validation accuracy
```

## Make it your own (5 ideas with the same stack)

1. **Yoga pose correction** — MediaPipe Pose instead of Hands
2. **Driver drowsiness detection** — eye landmarks + blink rate
3. **Plant leaf disease from phone camera** — same CNN training, new dataset
4. **Gesture volume / slide control** — open palm = next slide, fist = pause
5. **Urdu fingerspelling** — collect your own dataset, same pipeline, new letters

## For your portfolio / PhD applications

This project is complete and portfolio-ready: trained model (99% validation accuracy), live webcam demo verified, GitHub repository public, and a documented domain-shift lesson with a concrete fix. Next steps to strengthen it further: record the 30-second demo video, collect webcam data and retrain for higher live accuracy, and extend to system-wide typing (type in Notepad / Word directly from hand signs) and dynamic signs (words and sentences, not just letters).

Built by Hafiz Muhammad Ishaq — PyTorch / OpenCV / MediaPipe.
