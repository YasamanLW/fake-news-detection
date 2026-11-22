
# Fake News Detection Using Deep Learning

## Overview

This project implements a Deep Learning-based Fake News Detection system using Natural Language Processing (NLP) and PyTorch.

The primary objective is to classify news articles into two categories: Fake News and Real News based on their textual content.

The system uses TF-IDF for text feature extraction and a Feedforward Neural Network for binary classification.

## Objectives

- Process and prepare news article datasets.
- Extract meaningful textual features using TF-IDF.
- Build and train a neural network using PyTorch.
- Evaluate classification performance using standard metrics.
- Predict whether newly provided news articles are fake or real.

## Technologies & Libraries

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Natural Language Processing (NLP)
- TF-IDF Vectorization

## Project Workflow

1. Load Fake and Real news datasets.
2. Assign binary labels to the articles.
3. Combine and shuffle the datasets.
4. Split the data into training and testing sets.
5. Convert textual data into numerical features using TF-IDF.
6. Build a Feedforward Neural Network.
7. Train the model using the Adam optimizer and Binary Cross-Entropy loss.
8. Evaluate the model using classification metrics.
9. Test predictions on new news samples.

## Model Architecture

The neural network consists of:

- Input Layer: 2,000 features
- Hidden Layer 1: 128 neurons with ReLU activation
- Dropout Layer: 0.3
- Hidden Layer 2: 64 neurons with ReLU activation
- Dropout Layer: 0.3
- Output Layer: 1 neuron with Sigmoid activation

## Dataset

The project uses two CSV files:

- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

Both files should be placed inside the `data/` directory.

The CSV files must contain a `text` column.

Dataset files are not included in this repository.

## Experimental Results

The final mini-batch training experiment achieved the following results:

| Metric | Result |
|---|---:|
| Training Samples | 35,918 |
| Testing Samples | 8,980 |
| TF-IDF Features | 2,000 |
| Training Epochs | 10 |
| Test Accuracy | 98.67% |

### Confusion Matrix

| | Predicted Fake | Predicted Real |
|---|---:|---:|
| Actual Fake | 4,670 | 40 |
| Actual Real | 79 | 4,191 |

### Classification Performance

- Fake News F1-score: 0.99
- Real News F1-score: 0.99
- Overall Accuracy: 0.99 (rounded)

Note: These results are based on a random train-test split of the available dataset. Performance on news from entirely unseen sources may differ.

## Installation

Clone the repository:

```bash
git clone https://github.com/YasamanLW/fake-news-detection.git
```

Navigate to the project directory:

```bash
cd fake-news-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place `Fake.csv` and `True.csv` inside the `data/` directory.

Run the project:

```bash
python fake_news_detector.py
```

## Future Improvements

- Experiment with advanced NLP techniques.
- Evaluate transformer-based architectures such as BERT.
- Improve generalization across different news sources.
- Develop a web-based interface for real-time predictions.
- Add explainability techniques for model predictions.

## Author

**Yasaman Fallahi**

Computer Engineering | Artificial Intelligence

## License

This project is intended for educational and research purposes.
