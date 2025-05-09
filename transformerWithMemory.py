
from sklearn.model_selection import train_test_split
import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset


class NewsClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=2048):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True,
                                   max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def prepare_article_sequences(news_data, price_changes, max_articles_per_input=5):
    grouped = {}  # (ticker, date) → list of texts
    label_map = {}  # (ticker, date) → label

    for ticker, articles in news_data.items():
        for article in articles:
            aid = article.get("id")
            if not aid or ticker not in price_changes or aid not in price_changes[ticker]:
                continue

            dt = article.get("datetime")
            if not isinstance(dt, (int, float)):
                continue
            date = datetime.datetime.fromtimestamp(dt).date()

            key = (ticker, date)
            if key not in grouped:
                grouped[key] = []

            text = article.get("headline", "") + " " + article.get("summary", "")
            grouped[key].append(text.strip())

            # Save binary label (up=1, down=0)
            change = price_changes[ticker][aid]
            if change is None:
                continue
            label_map[key] = int(change > 0)

    texts, labels = [], []
    for key in grouped:
        articles = grouped[key][:max_articles_per_input]
        if len(articles) == 0 or key not in label_map:
            continue
        combined = " [SEP] ".join(articles)
        texts.append(combined)
        labels.append(label_map[key])

    return texts, labels


def train_longformer_classifier():
    print("Loading news...")
    news_data = loadCompanyNews()

    print("Calculating price changes...")
    price_changes = getStockPriceChange(news_data)

    print("Preparing article inputs...")
    texts, labels = prepare_article_sequences(news_data, price_changes)

    print(f"Total samples: {len(texts)}")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_dataset = NewsClassificationDataset(X_train, y_train, tokenizer)
    test_dataset = NewsClassificationDataset(X_test, y_test, tokenizer)

    model_path = "longformer_model.pkl"

    if os.path.exists(model_path):
        print("Loading model from pickle...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        print("Training new model...")
        model = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./longformer_cls_output",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=6,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            logging_dir="./logs",
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        def compute_metrics(pred):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            preds = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            return {
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds),
                'recall': recall_score(labels, preds),
                'f1': f1_score(labels, preds)
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.evaluate()

        print("Saving model with pickle...")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    train_longformer_classifier()