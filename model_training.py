import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("🟢 Training Script Started")

# Simulated dataset: [skin_tone, mouth_brightness, contrast]
data = [
    [180, 120, 20, 'United States', 'Happy'],
    [135, 70, 55, 'Indian', 'Neutral'],
    [85, 110, 40, 'African', 'Surprised'],
    [145, 100, 65, 'Indian', 'Happy'],
    [170, 130, 25, 'United States', 'Surprised'],
    [90, 90, 35, 'African', 'Neutral'],
    [160, 95, 60, 'Indian', 'Happy'],
    [100, 105, 50, 'African', 'Neutral'],
    [175, 140, 30, 'United States', 'Happy'],
    [125, 80, 45, 'Indian', 'Surprised'],
]

df = pd.DataFrame(data, columns=['skin_tone', 'mouth_brightness', 'contrast', 'nationality', 'emotion'])

X = df[['skin_tone', 'mouth_brightness', 'contrast']]
y_nat = df['nationality']
y_emo = df['emotion']

# -------- Train Nationality Model --------
X_train_nat, X_test_nat, y_train_nat, y_test_nat = train_test_split(X, y_nat, test_size=0.3, random_state=42)
model_nat = SVC(kernel="linear")
model_nat.fit(X_train_nat, y_train_nat)
y_pred_nat = model_nat.predict(X_test_nat)
print("\n🌍 Nationality Prediction Report:\n", classification_report(y_test_nat, y_pred_nat))

# -------- Train Emotion Model --------
X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(X, y_emo, test_size=0.3, random_state=42)
model_emo = SVC(kernel="linear")
model_emo.fit(X_train_emo, y_train_emo)
y_pred_emo = model_emo.predict(X_test_emo)
print("\n😊 Emotion Prediction Report:\n", classification_report(y_test_emo, y_pred_emo))

# -------- Save Models --------
os.makedirs("models", exist_ok=True)
joblib.dump(model_nat, "models/nationality_model.pkl")
print("✅ Nationality model saved at models/nationality_model.pkl")
joblib.dump(model_emo, "models/emotion_model.pkl")
print("✅ Emotion model saved at models/emotion_model.pkl")
