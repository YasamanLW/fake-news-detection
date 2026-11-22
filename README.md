
# Fake News Detection Using Deep Learning

A binary text classification system that detects fake and real news using TF-IDF feature extraction and a Feedforward Neural Network implemented with PyTorch.

## Project Overview

This project implements a deep learning approach for fake news detection. News articles are converted into numerical feature vectors using TF-IDF, then classified by a fully connected neural network.

The model is trained and evaluated on a dataset containing 44,898 news articles.

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- TF-IDF

## Dataset

The dataset consists of two CSV files:

- `Fake.csv`: Fake news articles (label 0)
- `True.csv`: Real news articles (label 1)

Dataset files are not included in this repository. Place both files inside the `data/` directory before running the project.

## Model Architecture

The neural network consists of:

1. Input layer: 2,000 TF-IDF features
2. Fully connected layer: 128 neurons + ReLU
3. Dropout: 0.3
4. Fully connected layer: 64 neurons + ReLU
5. Dropout: 0.3
6. Output layer: 1 neuron + Sigmoid

## Training Configuration

| Parameter | Value |
|---|---:|
| Train-test split | 80% / 20% |
| TF-IDF features | 2,000 |
| Batch size | 128 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Binary Cross Entropy |
| Random state | 42 |

## Experimental Results

The model was evaluated on 8,980 test samples.

**Test Accuracy: 98.69%**

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fake News | 0.98 | 0.99 | 0.99 |
| Real News | 0.99 | 0.98 | 0.99 |

### Confusion Matrix

| | Predicted Fake | Predicted Real |
|---|---:|---:|
| Actual Fake | 4668 | 42 |
| Actual Real | 76 | 4194 |

## Project Structure

```text
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── fake_news_model.pth
│   └── tfidf_vectorizer.joblib
│
├── fake_news_detector.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YasamanLW/fake-news-detection.git
cd fake-news-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Place `Fake.csv` and `True.csv` inside the `data/` directory.

Run:

```bash
python fake_news_detector.py
```

The script trains the model, evaluates its performance, saves the trained model and TF-IDF vectorizer, and performs a sample prediction.

## Limitations

- Performance is measured using a random train-test split of the available dataset.
- High test accuracy does not necessarily indicate reliable performance on unseen news sources or newly published articles.
- The model identifies patterns in text rather than independently verifying factual claims.
- Dataset quality and potential source overlap may affect evaluation results.

## Author

**Yasaman Fallahi**

GitHub: [YasamanLW](https://github.com/YasamanLW)
