# 🎭 EmotionSense

## Overview

**EmotionSense** is a multimodal emotion recognition system that leverages both text and audio inputs to detect human emotions. By integrating natural language processing and audio signal analysis, the system aims to provide accurate emotion classification, enhancing applications in human-computer interaction, sentiment analysis, and more.

---

## 🧠 Features

- **Text-Based Emotion Detection**: Utilizes natural language processing techniques to analyze textual data and identify underlying emotions.
- **Audio-Based Emotion Detection**: Processes audio inputs to detect emotional cues through voice modulation and tone analysis.
-  **Image-Based Emotion Detection**: Processes input images to detect emotions using models like Resnet and Swin Transformers.
- **Interactive Frontend**: Provides a user-friendly interface for inputting data and viewing results.

---

## 📁 Project Structure

```
EmotionSense/
├── attribute_resnet_model.ipynb   # Notebook for attribute-based ResNet model
├── audio.ipynb                    # Audio processing and analysis
├── audio_prediction.py            # Script for predicting emotions from audio
├── class_report.ipynb             # Generates classification reports(image)
├── cleaned_predict.ipynb          # Prediction with cleaned data
├── frontend.py                    # Frontend interface script
├── goemotions_1 (1).csv           # Dataset file (GoEmotions)
├── model.ipynb                    # Model training and evaluation for image
├── model2.ipynb                   # Alternative model experiments for image
├── text_prediction.py             # Script for predicting emotions from text
├── train_model.py                 # Model training script for text
└── README.md                      # Project documentation
```

---

## 📊 Dataset

The project utilizes the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions), a human-annotated dataset of 58k Reddit comments labeled for 27 emotion categories.

---

## 🛠️ Technologies Used

- **Programming Languages**: Python
- **Libraries & Frameworks**:
  - `pandas`, `numpy` for data manipulation
  - `scikit-learn` for machine learning algorithms
  - `librosa` for audio processing
  - `matplotlib`, `seaborn` for data visualization
  - `PyTorch` for deep learning models
  - `Flask` or `Streamlit` for the frontend interface

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akila-08/EmotionSense.git
   cd EmotionSense
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python frontend.py
   ```

---


---

## 📬 Contact

For any inquiries or feedback, please contact [akila-08](https://github.com/akila-08).

