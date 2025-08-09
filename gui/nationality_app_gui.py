import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

print("🟢 Script started")

# -------- Detection Logic Using OpenCV -------- #

def detect_skin_tone(image_path):
    img = cv2.imread(image_path)
    face = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(face, (0, 133, 77), (255, 173, 127))
    skin_pixels = cv2.bitwise_and(img, img, mask=skin_mask)

    skin_pixels_rgb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2RGB)
    skin_pixels_flat = skin_pixels_rgb.reshape(-1, 3)
    skin_pixels_flat = skin_pixels_flat[np.any(skin_pixels_flat != [0, 0, 0], axis=1)]

    if len(skin_pixels_flat) == 0:
        return "Other"

    avg_color = np.mean(skin_pixels_flat, axis=0)

    if avg_color[0] < 90:
        return "African"
    elif 90 <= avg_color[0] <= 150:
        return "Indian"
    elif avg_color[0] > 150:
        return "United States"
    else:
        return "Other"

def detect_dominant_color(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    lower_half = img[h//2:, :]
    lower_half = cv2.resize(lower_half, (100, 100))
    pixels = lower_half.reshape(-1, 3).astype(np.float32)

    _, _, centers = cv2.kmeans(pixels, 1, None, 
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                               10, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(map(int, centers[0]))

def detect_emotion(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Neutral"

    for (x, y, w, h) in faces:
        mouth_region = gray[y + int(0.65*h): y + h, x + int(0.2*w): x + int(0.8*w)]
        mouth_brightness = np.mean(mouth_region)

        if mouth_brightness < 80:
            return "Neutral"
        elif mouth_brightness < 110:
            return "Happy"
        else:
            return "Surprised"

    return "Neutral"

def estimate_age_category(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    if contrast > 70:
        return "Teen"
    elif 50 < contrast <= 70:
        return "Adult"
    elif 30 < contrast <= 50:
        return "Middle-aged"
    else:
        return "Senior"

# -------- GUI -------- #

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

        nationality = detect_skin_tone(file_path)
        emotion = detect_emotion(file_path)
        age_group = estimate_age_category(file_path)

        result = f"🌍 Nationality: {nationality}\n😊 Emotion: {emotion}\n🎂 Age Group: {age_group}"

        if nationality != "Indian":
            color = detect_dominant_color(file_path)
            result += f"\n👗 Dress Color (RGB): {color}"

        result_label.config(text=result)

# -------- GUI Window Setup -------- #

root = tk.Tk()
root.title("Nationality & Emotion Detector")
root.geometry("420x500")
root.config(bg="#f0f0f0")

tk.Label(root, text="Upload Face Image", font=("Helvetica", 14, "bold"), bg="#f0f0f0").pack(pady=10)
tk.Button(root, text="Browse Image", command=browse_image, font=("Helvetica", 12), bg="#007acc", fg="white").pack(pady=10)

image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f0f0", justify="left")
result_label.pack(pady=10)

print("🟢 GUI launching...")
root.mainloop()
