# SignFrame AI — Starter Code

Real-time American Sign Language (ASL) fingerspelling recognition with Python, PyTorch, OpenCV, and MediaPipe.

A custom model recognizes **24 static ASL handshapes** through a webcam, converts confirmed predictions into text, and speaks the word out loud.

> J and Z are excluded — they need motion, they aren't static shapes.

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

# 4. Train the landmark model (CPU is fine, ~10-20 min for full data)
python train_landmark.py --data data/asl_alphabet_train --epochs 30

# 5. Run it live
python inference.py --model models/landmark_mlp.pt
# Hold a sign steady ~1 second to confirm it.
# SPACE = speak, C = clear, BACKSPACE = delete, Q = quit
```

## Fixing domain shift (the most important lesson)

A model trained only on clean Kaggle photos will fail on your webcam — different lighting, background, camera, skin tone.

Do this:
1. `python collect_data.py --letter A --count 100` — repeat for each letter, in 2-3 different rooms.
2. Train with webcam data mixed in:
   ```bash
   python train_landmark.py --data data/asl_alphabet_train --webcam data/webcam --epochs 30
   ```
3. Test on your webcam, not just on Kaggle test data. Report both numbers:
   - Kaggle test accuracy: aim 95%+
   - Your webcam accuracy: aim 90%+

The landmark model already helps a lot: landmarks are normalized relative to the wrist, so hand position and size don't matter, and background is ignored completely.

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
    └── landmark_mlp.pt       # Trained model goes here
```

## Make it your own (5 ideas with the same stack)

1. **Yoga pose correction** — MediaPipe Pose instead of Hands
2. **Driver drowsiness detection** — eye landmarks + blink rate
3. **Plant leaf disease from phone camera** — same CNN training, new dataset
4. **Gesture volume / slide control** — open palm = next slide, fist = pause
5. **Urdu fingerspelling** — collect your own dataset, same pipeline, new letters

## For your portfolio / PhD applications

Add to your README when you finish:
- Demo video (30 sec, webcam + screen)
- Accuracy table: Kaggle test vs. your webcam test
- What you learned about domain shift, in 3 lines

Built as starter code for Hafiz Muhammad Ishaq — PyTorch / OpenCV / MediaPipe.
