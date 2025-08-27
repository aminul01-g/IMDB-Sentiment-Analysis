# IMDB Sentiment Analysis (Research-Focused)

## 💖 Abstract
This project investigates sentiment classification on the **IMDB Movie Reviews dataset**, a widely used benchmark for natural language processing (NLP) tasks. The primary objective is to build and evaluate machine learning models that classify reviews as **positive** or **negative**, and to analyze their effectiveness using both traditional and modern approaches.

---

## 📂 Dataset
- **Source**: [IMDB Dataset of 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)  
- **Size**: 50,000 reviews (25K for training, 25K for testing)  
- **Labels**:  
  - `0` → Negative sentiment  
  - `1` → Positive sentiment  

Preprocessing steps include:
- Tokenization  
- Lowercasing  
- Stopword removal  
- Lemmatization  
- Vectorization using **TF-IDF**  

---

## 🧠 Methodology
We experimented with both **traditional machine learning** and **deep learning** methods:

1. **Baseline Models**
   - Logistic Regression  
   - Naïve Bayes  
   - Support Vector Machines (SVM)  

2. **Deep Learning Models**
   - Artificial Neural Networks (ANN)  
   - Recurrent Neural Networks (RNN, LSTM)  

3. **Feature Engineering**
   - Bag-of-Words  
   - TF-IDF  
   - Word Embeddings (Word2Vec, GloVe, BERT)  

4. **Evaluation Metrics**
   - Accuracy  
   - Precision, Recall, F1-Score  
   - Confusion Matrix  

---

## 📊 Results

### Model Performance Comparison

| Method                  | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Logistic Regression (TF-IDF) | 88%     | 0.88      | 0.88   | 0.88     |
| SVM (TF-IDF)            | 89%     | 0.89      | 0.89   | 0.89     |
| LSTM (Word Embeddings)  | 91%     | 0.91      | 0.91   | 0.91     |
| BERT (CLS Embeddings)   | 93%     | 0.93      | 0.93   | 0.93     |

> *Note: Replace numbers with actual metrics from your notebook.*

---

### Visualizations

**1️⃣ Accuracy & Loss Curves**  
![Accuracy and Loss](./artifacts/accuracy_loss_plot.png)  
> *Line charts showing training/validation accuracy and loss per epoch.*

**2️⃣ Confusion Matrix (Best Model)**  
![Confusion Matrix](./artifacts/confusion_matrix.png)  
> *Displays true positives, true negatives, false positives, and false negatives.*

**3️⃣ Error Analysis**
- Misclassified examples often involve:
  - Mixed or nuanced sentiment: `"Great performances but a dull script"`  
  - Sarcasm or humor  
  - Domain-specific references (actors, genres)  

---

## 🌎 Discussion & Future Work
- **Strengths**:  
  - Simple preprocessing with TF-IDF yields strong baseline results.  
  - Deep learning models capture context and semantics better, improving F1-score.  

- **Limitations**:  
  - LSTM and BERT training are computationally expensive.  
  - Generalization to other domains (e.g., product reviews) is untested.  

- **Future Work**:  
  - Fine-tune transformer-based models (BERT, RoBERTa) for higher accuracy.  
  - Apply domain adaptation for other review datasets.  
  - Incorporate explainability tools (LIME, SHAP) to interpret predictions.  

---

## 📁 File Structure (Optional)
```
Module13_IMDB_Sentiment/
│
├─ Module13_IMDB_Sentiment_<YourName>.ipynb
├─ README.md
├─ artifacts/
│   ├─ accuracy_loss_plot.png
│   └─ confusion_matrix.png
└─ requirements.txt
```
> Replace `artifacts/*.png` with actual plots generated from your notebook.

---

## 📍 References
- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL.  
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.  
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.  

---

## 👌 Acknowledgments
This project was developed as part of **Module 13: IMDB Sentiment Analysis** in the AI/ML learning journey. Special thanks to the open-source community for providing tools and datasets that enabled this research.

