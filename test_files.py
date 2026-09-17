import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
import os
import random

# ---------------- MODEL DEFINITION (must match train_model.py) ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------- LOAD MODEL + NORM STATS ----------------
model = CNN()
model.load_state_dict(torch.load("voice_cnn_best.pt", map_location="cpu"))
model.eval()

norm_stats = joblib.load("cnn_norm_stats.pkl")
DATA_MEAN = norm_stats["mean"]
DATA_STD = norm_stats["std"]

print(f"Loaded model. mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}\n")


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :130]
    return mfcc


def predict_file(file_path):
    mfcc = extract_features(file_path)
    mfcc = (mfcc - DATA_MEAN) / (DATA_STD + 1e-8)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).reshape(1, 1, 40, 130)
    with torch.no_grad():
        pred = model(mfcc_tensor).item()
    return pred


# ---------------- TEST ON RANDOM SAMPLES FROM EACH FOLDER ----------------
real_files = [f for f in os.listdir("dataset/real") if f.endswith(".wav")]
fake_files = [f for f in os.listdir("dataset/fake") if f.endswith(".wav")]

random.shuffle(real_files)
random.shuffle(fake_files)

N = 10  # how many samples to test from each class

print(f"=== Testing {N} REAL files (expected output close to 0) ===")
real_correct = 0
for f in real_files[:N]:
    path = os.path.join("dataset/real", f)
    pred = predict_file(path)
    label = "REAL" if pred < 0.5 else "FAKE"
    correct = "✅" if pred < 0.5 else "❌"
    if pred < 0.5:
        real_correct += 1
    print(f"{correct} {f}: {pred:.4f} -> predicted {label}")

print(f"\nReal accuracy on this sample: {real_correct}/{N}\n")

print(f"=== Testing {N} FAKE files (expected output close to 1) ===")
fake_correct = 0
for f in fake_files[:N]:
    path = os.path.join("dataset/fake", f)
    pred = predict_file(path)
    label = "REAL" if pred < 0.5 else "FAKE"
    correct = "✅" if pred >= 0.5 else "❌"
    if pred >= 0.5:
        fake_correct += 1
    print(f"{correct} {f}: {pred:.4f} -> predicted {label}")

print(f"\nFake accuracy on this sample: {fake_correct}/{N}")
