"""
SignFrame AI - MediaPipe hand helpers
Shared by collect_data.py, train_landmark.py and inference.py
"""
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_hand_detector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def landmarks_to_vector(hand_landmarks):
    """
    Convert 21 landmarks -> 63 numbers, normalized relative to wrist.
    Normalization is the key trick against domain shift:
    hand position / size in the frame no longer matters.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist  # translate so wrist is at (0,0,0)
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts = pts / scale  # scale so hand size is ~1
    return pts.flatten()  # shape (63,)


def extract_landmarks_from_image(hands, image_bgr):
    """Return 63-vector or None if no hand found. image_bgr = OpenCV BGR image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None
    lm = result.multi_hand_landmarks[0]
    return landmarks_to_vector(lm), lm


def crop_hand(image_bgr, hand_landmarks, padding=30):
    """Crop a square box around the hand for the image CNN."""
    h, w, _ = image_bgr.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]
