# Data folder

## 1. Kaggle ASL Alphabet Dataset (87,000 images)
Download from Kaggle: search "ASL Alphabet"
Unzip here so the layout is:
```
data/asl_alphabet_train/A/0001.jpg
data/asl_alphabet_train/B/0001.jpg
...
```
J and Z folders are ignored by training (they need motion).

## 2. Your own webcam data
Collected with:
```
python collect_data.py --letter A --count 100
```
Goes to `data/webcam/A/0001.jpg` + `0001.npy` (landmarks).
Collect in 2-3 different rooms / lightings.
