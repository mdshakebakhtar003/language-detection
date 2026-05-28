# 🌐 Language Detection — NLP Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

A machine learning project that detects the language of any input text using **Bag-of-Words** vectorization and three classification models — **Multinomial Naive Bayes**, **Logistic Regression**, and **Random Forest**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models Used](#-models-used)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Sample Output](#-sample-output)
- [Future Improvements](#-future-improvements)

---

## 📖 Overview

This project builds a **text classification pipeline** that identifies which language a given sentence or paragraph is written in. It supports **22 languages** and is trained on a balanced dataset of 22,000 text samples.

The pipeline uses:
- `CountVectorizer` to convert raw text into numerical features (Bag of Words)
- Three ML classifiers to compare performance
- A live interactive prediction loop in the terminal

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| File | `language.csv` |
| Total Rows | 22,000 |
| Columns | `Text`, `language` |
| Languages | 22 |
| Samples per language | 1,000 (perfectly balanced) |

### Supported Languages

| | | | | |
|--|--|--|--|--|
| Estonian | Swedish | English | Russian | Romanian |
| Persian | Pushto | Spanish | Hindi | Korean |
| Chinese | French | Portugese | Indonesian | Urdu |
| Latin | Turkish | Japanese | Dutch | Tamil |
| Thai | Arabic | | | |

---

## 📁 Project Structure

```
language-detection/
│
├── language_detection.py   # Main ML script
├── language.csv            # Dataset
└── README.md               # This file
```

---

## 🤖 Models Used

### 1. Multinomial Naive Bayes
- Fast, lightweight probabilistic classifier
- Works very well with word count features
- Best suited for text classification baselines

### 2. Logistic Regression
- Linear classifier with strong generalization
- Handles high-dimensional sparse data well
- Usually achieves highest accuracy on this task

### 3. Random Forest
- Ensemble of 100 decision trees
- Robust to overfitting
- Slower to train but captures non-linear patterns

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | ~95.32% |
| Logistic Regression | ~97–98% ⭐ |
| Random Forest | ~95–96% |

> ⭐ Logistic Regression typically achieves the best accuracy on this dataset.

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/language-detection.git
cd language-detection
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn
```

> Python 3.8 or higher is recommended.

---

## 🚀 Usage

```bash
python language_detection.py
```

The script will:
1. Load and explore the dataset
2. Train all 3 models
3. Print accuracy and classification report for each
4. Show a comparison table with the best model highlighted
5. Start an interactive prediction loop

**In the prediction loop:**
```
Enter a Text: 내 이름은 스와티입니다
  [MultinomialNB]
  Detected Language : Korean
  Confidence        : 98.45%

  Top-3 Predictions:
    Korean           98.4%  ███████████████████
    Japanese          1.2%  
    Chinese           0.4%  
```

Type `switch` to change the active model, or `quit` to exit.

---

## 🔍 How It Works

```
Raw Text
   │
   ▼
CountVectorizer  ──►  Bag-of-Words Matrix  (word frequency counts)
   │
   ▼
Train/Test Split  (67% train | 33% test)
   │
   ├──► MultinomialNB       ──► Accuracy ~95.32%
   ├──► LogisticRegression  ──► Accuracy ~97–98%
   └──► RandomForest        ──► Accuracy ~95–96%
   │
   ▼
Best Model Selected  ──►  Live Prediction
```

**CountVectorizer** converts text like:
```
"I love data science"  →  {'data': 1, 'love': 1, 'science': 1}
```
into a numeric vector the model can learn from.

---

## 💻 Sample Output

```
============================================================
        LANGUAGE DETECTION - NLP PROJECT
============================================================

[1] Dataset Loaded Successfully
    Shape : (22000, 2)

[3] Language Distribution:
    Estonian      1000
    Swedish       1000
    ...

[7] Training Models...
    Training MultinomialNB          Done ✓
    Training LogisticRegression     Done ✓
    Training RandomForestClassifier Done ✓

============================================================
  ACCURACY SUMMARY
============================================================
  MultinomialNB              95.32%  ██████████████████████████████████████
  LogisticRegression         97.81%  ███████████████████████████████████████  ◀ BEST
  RandomForestClassifier     95.74%  ██████████████████████████████████████

  Best Model : LogisticRegression  (97.81%)
============================================================
```

---

## 🔮 Future Improvements

- [ ] Use `TfidfVectorizer` instead of `CountVectorizer` for better weighting
- [ ] Add character-level n-grams `(analyzer='char_wb', ngram_range=(2,4))`
- [ ] Try `LinearSVC` for potentially 98–99% accuracy
- [ ] Add text preprocessing (lowercasing, removing noise)
- [ ] Save the best model using `joblib` for deployment
- [ ] Build a web app with Flask or Streamlit
- [ ] Add support for more languages

---

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**MD Shakeb**
- GitHub: [@your-username](https://github.com/mdshakebakhtar003)

---

> Made with ❤️ as part of an NLP / Machine Learning portfolio project.
