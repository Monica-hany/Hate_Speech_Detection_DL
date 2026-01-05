# 🛡️ Hate Speech Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Ain Shams University - Faculty of Engineering**  
**CSE485 Deep Learning: Major Task**

---

## 📋 Project Overview

This project implements a **Deep Learning model** to detect hate speech in tweets using **LSTM (Long Short-Term Memory)** networks. The model classifies tweets into three categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Hate Speech | Content explicitly targeting individuals or groups with harmful intent |
| 1 | Offensive Language | Content containing offensive language but not necessarily hate speech |
| 2 | Neither | Neutral content without any offensive or hateful intent |

## 📌 Important Note

Among all notebooks in this repository, the **final and complete implementation** of the project is provided in the notebook:

**`Team_5_DL_Project.ipynb`**

All experiments, preprocessing steps, model training, evaluation, and final results are consolidated in this notebook. Other notebooks are intermediate or exploratory and are included for reference only.


## 🎯 Objectives

- ✅ Explore and visualize the dataset
- ✅ Preprocess textual data (stopwords, punctuation removal, lemmatization)
- ✅ Handle class imbalance using oversampling
- ✅ Build and train LSTM models
- ✅ Compare multiple model architectures
- ✅ Evaluate model performance
- ✅ Analyze results comprehensively

## 📊 Dataset

- **Source**: Hate Speech Dataset
- **Size**: 24,783 tweets
- **Columns**: `tweet` (text), `class` (label)
- **Class Distribution**:
  - Offensive Language: 77.4% (19,190 tweets)
  - Neither: 16.8% (4,163 tweets)
  - Hate Speech: 5.8% (1,430 tweets)

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep Learning framework
- **NLTK** - Natural Language Processing
- **Scikit-learn** - Model evaluation & utilities
- **Imbalanced-learn** - SMOTE for handling class imbalance
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

## 🛠️ Requirements

pandas
numpy
matplotlib
seaborn
nltk
tensorflow>=2.0
scikit-learn
imbalanced-learn
gdown


## 🧠 Model Architectures

### Model 1 (High Capacity)
Embedding (32) → Bidirectional LSTM (16) → Dense (512) → BatchNorm → Dropout (0.3) → Softmax (3)

### Model 2 (Regularized) - **Recommended**
Embedding (32) → SpatialDropout (0.3) → Bidirectional LSTM (8) → Dense (32) → BatchNorm → Dropout (0.5) → Softmax (3)


---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 5,000 |
| Max Sequence Length | 30 |
| Embedding Dimension | 32 |
| LSTM Units | 8 (Model 2), 16 (Model 1) |
| Dropout Rate | 0.3 - 0.5 |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Optimizer | Adam |

---

## 📈 Results

### Model Comparison

| Metric | Model 1 | Model 2 |
|--------|---------|---------|
| **Validation Accuracy** | 88% | 87% |
| **Validation Loss** | 0.63 | 0.40 |
| **Epochs to Converge** | 15 | 8 |
| **Overfitting** | Significant | Well-controlled |

### Classification Report (Model 2)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Hate Speech (0) | 0.85 | 0.90 | 0.87 |
| Offensive (1) | 0.87 | 0.82 | 0.84 |
| Neither (2) | 0.88 | 0.90 | 0.89 |

### Key Findings

**Model 2 is recommended** for deployment due to:
- ✅ 40% lower validation loss (better confidence)
- ✅ Less overfitting (better generalization)
- ✅ Faster convergence (8 epochs vs 15)
- ✅ Smaller model size (faster inference)

---

## 📝 Text Preprocessing Pipeline

1. Remove user mentions (@username)
2. Remove HTML entities
3. Remove URLs
4. Remove punctuation and special characters
5. Remove stopwords
6. Lemmatization

## ⚠️ Limitations & Improvements

### 🔹 Stratified Data Splitting (Missing Improvement)

One limitation of the current implementation is **not using stratified splitting** when dividing the dataset into training and validation sets.

**Why stratification is important:**
- The dataset is **highly imbalanced**, especially for the *Hate Speech* class
- Without stratification, some splits may contain **very few samples of minority classes**
- This can lead to **biased evaluation** and **unstable performance metrics**

**Improvement:**  
Using **stratified train-validation splitting** ensures that each split preserves the **original class distribution**, leading to:
- More reliable evaluation
- Better generalization


## 📝 License

This project is for educational purposes as part of CSE485 Deep Learning course at Ain Shams University.

---

## 🙏 Acknowledgments

- Ain Shams University - Faculty of Engineering
- CSE485 Deep Learning Course Instructors

---
