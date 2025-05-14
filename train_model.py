import pandas as pd
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
)
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    emotion_labels = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]

    df = df[df["example_very_unclear"] == False]
    df["labels"] = df[emotion_labels].values.tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["labels"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df[emotion_labels].idxmax(axis=1),
    )

    return train_texts, test_texts, train_labels, test_labels, emotion_labels


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.FloatTensor(self.labels[item]),
        }


def compute_class_weights(labels):
    all_labels = np.concatenate(labels)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(all_labels), y=all_labels
    )
    return torch.tensor(class_weights, dtype=torch.float)


def train_model(train_texts, train_labels, emotion_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(emotion_labels)},
        label2id={label: i for i, label in enumerate(emotion_labels)},
    ).to(device)

    class_weights = compute_class_weights(train_labels)

    def weighted_loss(outputs, labels):
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        return loss_fct(outputs.logits, labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        logging_steps=50,
        seed=42,
    )

    def compute_metrics(p: EvalPrediction):
        preds = (p.predictions > 0).astype(int)
        return {
            "micro_f1": f1_score(p.label_ids, preds, average="micro"),
            "macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "accuracy": accuracy_score(p.label_ids, preds),
        }

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    eval_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    return model, tokenizer


def evaluate_model(model, tokenizer, test_texts, test_labels, emotion_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    model.eval()
    predictions = []
    with torch.no_grad():
        for item in test_dataset:
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            }
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions).squeeze()
    true_labels = np.array(test_labels)

    optimal_thresholds = []
    for i in range(len(emotion_labels)):
        precision, recall, thresholds = precision_recall_curve(
            true_labels[:, i], predictions[:, i]
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_thresholds.append(thresholds[np.argmax(f1_scores)])

    pred_labels = np.zeros_like(predictions)
    for i, thresh in enumerate(optimal_thresholds):
        pred_labels[:, i] = (predictions[:, i] > thresh).astype(int)

    print("Enhanced Classification Report:")
    print(
        classification_report(
            true_labels, pred_labels, target_names=emotion_labels, zero_division=0
        )
    )

    print("\nClass-wise Optimal Thresholds:")
    for label, thresh in zip(emotion_labels, optimal_thresholds):
        print(f"{label}: {thresh:.3f}")

    print("\nConfusion Matrix for Top 5 Emotions:")
    top_emotions = np.argsort(-true_labels.sum(axis=0))[:5]
    for i in top_emotions:
        cm = confusion_matrix(true_labels[:, i], pred_labels[:, i])
        print(f"\n{emotion_labels[i]}:")
        print(cm)

    return predictions, optimal_thresholds


if __name__ == "__main__":
    train_texts, test_texts, train_labels, test_labels, emotion_labels = (
        load_and_prepare_data(
            r"C:\Users\pssan\OneDrive\Desktop\New_customer\goemotions_1.csv"
        )
    )

    model, tokenizer = train_model(train_texts, train_labels, emotion_labels)

    model.save_pretrained("./enhanced_emotion_model")
    tokenizer.save_pretrained("./enhanced_emotion_model")

    print("\nEvaluating on test set...")
    predictions, optimal_thresholds = evaluate_model(
        model, tokenizer, test_texts, test_labels, emotion_labels
    )

    print("\nSample Predictions with Optimal Thresholds:")
    for i in range(5):
        print(f"\nText: {test_texts[i]}")
        print(
            "True:",
            [emotion_labels[j] for j, val in enumerate(test_labels[i]) if val == 1],
        )
        print(
            "Pred:",
            [
                emotion_labels[j]
                for j, val in enumerate(
                    (predictions[i] > np.array(optimal_thresholds)).astype(int)
                )
                if val == 1
            ],
        )
        print("Top Probabilities:")
        top_indices = np.argsort(predictions[i])[::-1][:3]
        for idx in top_indices:
            print(
                f"  {emotion_labels[idx]}: {predictions[i][idx]:.3f} (threshold: {optimal_thresholds[idx]:.3f})"
            )
