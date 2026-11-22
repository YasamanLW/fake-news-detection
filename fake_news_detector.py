
from pathlib import Path

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FAKE_FILE = DATA_DIR / "Fake.csv"
TRUE_FILE = DATA_DIR / "True.csv"

RANDOM_STATE = 42
MAX_FEATURES = 2000
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# 3. Load Dataset
# ============================================================

def load_dataset():
    if not FAKE_FILE.exists() or not TRUE_FILE.exists():
        raise FileNotFoundError(
            "Dataset files were not found. "
            "Place Fake.csv and True.csv inside the data folder."
        )

    fake_data = pd.read_csv(FAKE_FILE)
    true_data = pd.read_csv(TRUE_FILE)

    if "text" not in fake_data.columns or "text" not in true_data.columns:
        raise ValueError("Both CSV files must contain a 'text' column.")

    fake_data["label"] = 0
    true_data["label"] = 1

    data = pd.concat(
        [fake_data, true_data],
        ignore_index=True
    )

    data = data.dropna(subset=["text"])
    data = data.sample(frac=1, random_state=RANDOM_STATE)
    data = data.reset_index(drop=True)

    return data


# ============================================================
# 4. Neural Network Architecture
# ============================================================

class FakeNewsNN(nn.Module):

    def __init__(self, input_size):
        super(FakeNewsNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 5. Train Model
# ============================================================

def train_model():

    print("\nLoading dataset...")

    data = load_dataset()

    X = data["text"]
    y = data["label"]

    print(f"Total samples: {len(data)}")
    print(f"Fake news samples: {(y == 0).sum()}")
    print(f"Real news samples: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print("\nExtracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_tensor = torch.tensor(
        X_train_tfidf.toarray(),
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_tfidf.toarray(),
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = FakeNewsNN(MAX_FEATURES).to(DEVICE)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting model training...\n")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Loss: {average_loss:.4f}"
        )

    # ========================================================
    # 6. Model Evaluation
    # ========================================================

    model.eval()

    with torch.no_grad():

        X_test_tensor = X_test_tensor.to(DEVICE)

        predictions = model(X_test_tensor)

        predicted_labels = (
            predictions >= 0.5
        ).int().cpu().numpy().flatten()

    y_true = y_test.to_numpy()

    accuracy = accuracy_score(
        y_true,
        predicted_labels
    )

    print("\nModel Evaluation")
    print("=" * 50)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predicted_labels,
            target_names=["Fake News", "Real News"]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predicted_labels))

    # ========================================================
    # 7. Save Model and Vectorizer
    # ========================================================

    MODEL_DIR.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_DIR / "fake_news_model.pth"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    print("\nModel and vectorizer saved successfully.")

    return model, vectorizer


# ============================================================
# 8. Predict News
# ============================================================

def predict_news(text, model, vectorizer):

    model.eval()

    text_vector = vectorizer.transform([text])

    text_tensor = torch.tensor(
        text_vector.toarray(),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        probability = model(text_tensor).item()

    prediction = (
        "Real News" if probability >= 0.5
        else "Fake News"
    )

    confidence = (
        probability if probability >= 0.5
        else 1 - probability
    )

    print("\nPrediction Result")
    print("-" * 30)
    print(f"Prediction: {prediction}")
    print(f"Real News Probability: {probability:.4f}")
    print(f"Confidence: {confidence:.2%}")

    return prediction, probability


# ============================================================
# 9. Main Execution
# ============================================================

if __name__ == "__main__":

    print("Fake News Detection System")
    print("=" * 50)
    print(f"Running on: {DEVICE}")

    model, vectorizer = train_model()

    sample_news = (
        "The government announced a new economic policy "
        "during a press conference today."
    )

    predict_news(
        sample_news,
        model,
        vectorizer
    )
