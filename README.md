# Nationality & Emotion Detector

This project is a **GUI-based application** that detects a person's nationality and emotion from an image.  
It also provides additional information based on the nationality detected (Age estimation for Indian nationals, dominant dress color for others).

## 📌 Features
- Detects **Nationality** (Indian, African, United States, Other)
- Detects **Emotion** (Happy, Neutral, Surprised)
- Estimates **Age** for Indian nationals
- Detects **Dominant Dress Color** for non-Indian nationals
- Simple **Tkinter GUI** to upload and analyze images
- Uses **OpenCV** for image processing

## 🛠️ Technologies Used
- Python
- OpenCV
- Tkinter
- NumPy
- PIL (Pillow)
- Scikit-learn (for model training)
- Matplotlib, Seaborn (for analysis & visualization)

## 📂 Project Structure
```
├── gui
│   └── nationality_app_gui.py   # Main GUI application
├── models
│   ├── nationality_model.pkl    # Trained nationality classifier
│   └── emotion_model.pkl        # Trained emotion classifier
├── model_training.py            # Script to train and save models
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

## 🚀 How to Run
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/nationality-emotion-detector.git
   cd nationality-emotion-detector
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI application**  
   ```bash
   python gui/nationality_app_gui.py
   ```

4. **(Optional) Train the models again**  
   ```bash
   python model_training.py
   ```

## 📊 Model Training
- Dataset: Custom / Generated sample data
- Models: **SVM** classifiers for nationality & emotion
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
- Models achieve **≥70% accuracy**

## 📷 GUI Preview
*(Add screenshot of your GUI here)*

## 📄 License
This project is for **educational purposes** only.
