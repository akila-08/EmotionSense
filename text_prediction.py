from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np


def predict_emotions(text):
    
    model_path = "D:/sem 6/ai/package/text/enhanced_emotion_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    optimal_thresholds = [0.5] * len(emotion_labels)  

    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    pred_labels = (probs > np.array(optimal_thresholds)).astype(int)

    predicted_emotions = [emotion_labels[i] for i, val in enumerate(pred_labels) if val == 1]
    return predicted_emotions, probs





