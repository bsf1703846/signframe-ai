"""
SignFrame AI - Train the image CNN (STEP 2, after landmark model works)

Uses transfer learning (MobileNetV2) - same idea as your VGG-16 rice work.
Input: hand crop images, 128x128. Output: 24 letters.

Usage:
  python train_image.py --data data/asl_alphabet_train --epochs 15 --model mobilenet

  --model mobilenet  (recommended, pretrained, fast)
  --model cnn        (small custom CNN from scratch)

Data layout: ImageFolder format
  data/asl_alphabet_train/A/*.jpg, B/*.jpg, ... (J and Z ignored)
"""
import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import LETTERS_24, SimpleCNN, get_mobilenet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/asl_alphabet_train")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--model", default="mobilenet", choices=["mobilenet", "cnn"])
    p.add_argument("--out", default="models/image_model.pt")
    args = p.parse_args()

    # Augmentation is the domain-shift defence for the image model
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomResizedCrop(128, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Only keep the 24 static letters - filter ImageFolder classes
    full = datasets.ImageFolder(args.data, transform=train_tf)
    keep = [i for i, c in enumerate(full.classes) if c in LETTERS_24]
    print(f"Classes in folder: {full.classes}")
    print(f"Keeping 24 static letters, dropping J/Z and extras: {[full.classes[i] for i in keep]}")

    from torch.utils.data import Subset
    import numpy as np
    idx = [i for i, (_, label) in enumerate(full.samples) if label in keep]
    # Remap labels 0..23
    remap = {old: new for new, old in enumerate(sorted(keep))}
    # Simpler: use a wrapper dataset that remaps on the fly
    class Remapped(torch.utils.data.Dataset):
        def __init__(self, base, indices, remap, tf):
            self.base, self.indices, self.remap, self.tf = base, indices, remap, tf
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            path, label = self.base.samples[self.indices[i]]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return self.tf(img), self.remap[label]

    # Split train/val
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(len(idx) * 0.85)
    train_ds = Remapped(full, idx[:split], remap, train_tf)
    val_ds = Remapped(full, idx[split:], remap, val_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Train: {len(train_ds)}  Val: {len(val_ds)}")
    model = get_mobilenet() if args.model == "mobilenet" else SimpleCNN()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                truths.extend(yb.numpy())
        acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch+1}/{args.epochs}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "letters": LETTERS_24,
                        "model": args.model, "val_acc": float(acc)}, args.out)
            print(f"  -> saved best to {args.out}")
    print(f"Best val accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
