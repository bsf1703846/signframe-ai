"""
SignFrame AI - Train the landmark model (START HERE)

Pipeline:
  Kaggle ASL images (or your data/webcam images)
    -> MediaPipe extracts 21 landmarks per image
    -> 63 numbers per image
    -> MLP 63 -> 128 -> 64 -> 24
    -> models/landmark_mlp.pt

Usage:
  1. Download Kaggle ASL Alphabet Dataset, unzip to data/asl_alphabet_train/
     Folder structure expected:
       data/asl_alphabet_train/A/0001.jpg ...
       data/asl_alphabet_train/B/0001.jpg ...
     (J and Z folders are ignored - they need motion)

  2. python train_landmark.py --data data/asl_alphabet_train --epochs 30
     Add --webcam data/webcam to also mix in your own webcam photos.

  3. Test: python train_landmark.py --test-only --model models/landmark_mlp.pt

If you have no Kaggle data yet, run the smoke test:
  python train_landmark.py --smoke-test
  (trains 1 epoch on random synthetic landmarks, just to check the code runs)
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.hand_utils import get_hand_detector, extract_landmarks_from_image
from src.model import LETTERS_24, LETTER_TO_IDX, IDX_TO_LETTER, LandmarkMLP


def build_dataset(data_dirs):
    """Extract landmarks from all images in data_dirs. Returns X (N,63), y (N,)."""
    hands = get_hand_detector()
    # For still images, use static mode for better detection
    import mediapipe as mp
    hands_static = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.5,
    )
    X, y = [], []
    for data_dir in data_dirs:
        for letter in LETTERS_24:
            folder = os.path.join(data_dir, letter)
            if not os.path.isdir(folder):
                continue
            # .jpg images + pre-saved .npy landmarks from collect_data.py
            for npy_path in glob.glob(os.path.join(folder, "*.npy")):
                try:
                    vec = np.load(npy_path)
                    if vec.shape == (63,):
                        X.append(vec); y.append(LETTER_TO_IDX[letter])
                except Exception:
                    pass
            images = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            for img_path in tqdm(images, desc=f"{data_dir}/{letter}", leave=False):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                vec, _ = extract_landmarks_from_image(hands_static, img)
                if vec is not None:
                    X.append(vec); y.append(LETTER_TO_IDX[letter])
    hands_static.close()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train(X, y, epochs=30, batch_size=64, lr=0.001, model_out="models/landmark_mlp.pt"):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Classes: {len(set(y))}")

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = LandmarkMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        # val
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                truths.extend(yb.numpy())
        acc = accuracy_score(truths, preds)
        sched.step(1 - acc)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_dl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "letters": LETTERS_24, "val_acc": float(acc)}, model_out)
            print(f"  -> saved best model ({acc:.4f}) to {model_out}")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(classification_report(truths, preds, target_names=LETTERS_24, zero_division=0))
    return best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/asl_alphabet_train")
    p.add_argument("--webcam", default="data/webcam")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--model-out", default="models/landmark_mlp.pt")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        print("Smoke test: random synthetic landmarks, 1 epoch")
        X = np.random.randn(480, 63).astype(np.float32)
        y = np.random.randint(0, 24, size=480, dtype=np.int64)
        train(X, y, epochs=1, model_out="models/smoke_test.pt")
        print("Smoke test PASSED - training code works.")
        return

    dirs = [args.data]
    if os.path.isdir(args.webcam):
        dirs.append(args.webcam)
        print(f"Including webcam data from {args.webcam} (fixes domain shift)")
    X, y = build_dataset(dirs)
    if len(X) == 0:
        print("No data found. Download Kaggle ASL Alphabet to data/asl_alphabet_train/")
        print("Or collect your own: python collect_data.py --letter A --count 100")
        return
    print(f"Dataset: {len(X)} samples, {len(set(y))} classes")
    train(X, y, epochs=args.epochs, model_out=args.model_out)


if __name__ == "__main__":
    main()
