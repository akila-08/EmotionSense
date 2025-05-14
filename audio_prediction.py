from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa

def predict_emotion(file_path):
    
    model_path = "D:/sem 6/ai/package/audio/wav2vec2-crema-emotion"  
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    emotion_dict = {
    "SAD": 0, "ANG": 1, "DIS": 2, "FEA": 3, "HAP": 4, "NEU": 5
    }
    id2label = {v: k for k, v in emotion_dict.items()}   
    
    audio, sr = librosa.load(file_path, sr=16000)    
   
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        padding="max_length",
        max_length=160000, 
        truncation=True,
        return_tensors="pt"
    )
          
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
   
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs).item()
    
    return {
        "file": file_path,
        "prediction": id2label[pred_class],
        "confidence": probs[0][pred_class].item(),
        "all_probs": {id2label[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
    }
    

