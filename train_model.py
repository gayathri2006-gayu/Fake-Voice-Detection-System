import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib

print("CNN TRAINING STARTED")

real_dir = "dataset/real"
fake_dir = "dataset/fake"

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :130]
    return mfcc

print("Loading real audio...")
for file in os.listdir(real_dir):
    try:
        X.append(extract_features(os.path.join(real_dir, file)))
        y.append(0)
    except:
        pass

print("Loading fake audio...")
for file in os.listdir(fake_dir):
    try:
        X.append(extract_features(os.path.join(fake_dir, file)))
        y.append(1)
    except:
        pass

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

data_mean = np.mean(X)
data_std = np.std(X)

X = (X - data_mean) / (data_std + 1e-8)
X = X.reshape(X.shape[0], 1, 40, 130)

# Save the normalization values so detect.py can use the EXACT same scale
joblib.dump({"mean": data_mean, "std": data_std}, "cnn_norm_stats.pkl")
print("Saved normalization stats: mean =", data_mean, "std =", data_std)

print("Real samples:", np.sum(y == 0))
print("Fake samples:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

BATCH_SIZE = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

model = CNN()

criterion = nn.BCELoss()

# ---------------- CHANGE: lower learning rate ----------------
optimizer = optim.Adam(model.parameters(), lr=0.0002)

print("Training CNN...")

NUM_EPOCHS = 150

# ---------------- CHANGE: early stopping tracking ----------------
best_val_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Run validation EVERY epoch now, so we never miss the best one
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test).squeeze()
        val_preds_binary = (val_preds > 0.5).float()
        val_acc = (val_preds_binary == y_test).float().mean().item() * 100

    # Save checkpoint whenever we beat the previous best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "voice_cnn_best.pt")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss:.4f} | Val Accuracy: {val_acc:.2f}% | Best so far: {best_val_acc:.2f}% (epoch {best_epoch})")

print(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
print("Best model saved as voice_cnn_best.pt")

# Also save the final-epoch model for reference
torch.save(model.state_dict(), "voice_cnn_final.pt")
print("Final-epoch model saved as voice_cnn_final.pt")

print("CNN MODEL SAVED")