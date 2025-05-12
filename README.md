# End-to-End-Credit-Card-Fraud-Detection-with-Machine-Learning-Models

Built and benchmarked machine learning models to detect fraudulent credit card transactions in an imbalanced dataset. Applied EDA, NearMiss undersampling, and tuned Logistic Regression, KNN, Decision Tree, Random Forest, and AdaBoost. Logistic Regression and Decision Tree achieved 94% accuracy.


## 📌 Project Overview

This project applies machine learning techniques to detect fraudulent credit card transactions.  
Using the **Kaggle Credit Card Fraud Detection dataset**, the project benchmarks and compares the performance of multiple classification models under the challenge of extreme class imbalance.

---

## 📊 Dataset Overview

- **Source**: [Kaggle — Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions**: 284,807
- **Features**: 30 (28 PCA components + Time + Amount)
- **Target Variable**: `Class` (0 = Non-fraud, 1 = Fraud)
- **Fraud Cases**: ~0.17% (492 frauds)

| Feature | Description |
|---------|-------------|
| **Time** | Seconds elapsed between the transaction and the first transaction |
| **V1–V28** | PCA-transformed features (due to confidentiality) |
| **Amount** | Transaction amount |
| **Class** | Target variable (0 = Non-fraud, 1 = Fraud) |

---

## 🎯 Problem Statement

Fraudulent transactions result in significant financial losses and customer dissatisfaction for financial institutions.  
The aim is to build machine learning models that can **accurately detect fraudulent transactions** while handling **extreme class imbalance**, minimizing false negatives, and improving fraud detection efficiency.

---

## 🏆 Project Objectives

- Perform **exploratory data analysis (EDA)** to understand feature relationships and class distribution
- Address **class imbalance** using undersampling (NearMiss technique)
- Train and evaluate multiple **classification models**
- Optimize models through **hyperparameter tuning**
- Compare model performance and identify the most effective classifier

---

## ⚙️ Methodology

1. **Data Preprocessing**  
   - **NearMiss undersampling** to balance classes  
   - Feature scaling (StandardScaler) for distance-based models

2. **Model Building**  
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - AdaBoost Classifier

3. **Model Evaluation**  
   - Metrics: **Accuracy**, **Balanced Accuracy**, **Confusion Matrix**, **Classification Report**  
   - Applied **GridSearchCV** for hyperparameter tuning

4. **Model Comparison & Interpretation**  
   - Compared model results visually and numerically  
   - Analyzed feature importances

5. **Conclusion and Recommendations**

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn**  
- **Scikit-learn**  
- **Jupyter Notebook**

---

## 📈 Key Results

| Model | Test Accuracy (Post Tuning) |
|-------|-----------------------------|
| Logistic Regression | 94% |
| Decision Tree Classifier | 94% |
| Random Forest Classifier | 93% |
| K-Nearest Neighbors | 93% |
| AdaBoost Classifier | 83% |

✅ Logistic Regression and Decision Tree Classifier were the **top-performing models**  
✅ Hyperparameter tuning improved performance across models

---

## 📚 Conclusion

The project successfully benchmarked multiple machine learning models on a real-world fraud detection dataset.  
**Logistic Regression** and **Decision Tree Classifier** emerged as top performers, achieving **94% test accuracy** on the balanced dataset.  
The methodology of **undersampling + model tuning** effectively addressed the class imbalance challenge.

---

## 🙌 Acknowledgements

Dataset provided by **Machine Learning Group — Université Libre de Bruxelles**, available publicly on **Kaggle**.

---

## ✍️ Author

**Sharada Sonaje**  
_Machine Learning & Data Science Enthusiast_

- [LinkedIn](https://www.linkedin.com/) _(Add your link here)_  
- [Portfolio](https://github.com/) _(Add your portfolio link here)_  
