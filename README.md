# CreditCardFraud_ML
# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning. The dataset is highly imbalanced, where fraudulent transactions represent only a very small percentage of total transactions.

The goal of this project is to build a classification model that can accurately identify fraudulent transactions while minimizing false positives.

---

## 📊 Dataset

The dataset used in this project contains credit card transactions made by European cardholders.

Features include:

* **Time** – Seconds elapsed between transactions
* **Amount** – Transaction amount
* **V1–V28** – PCA transformed features
* **Class** – Target variable

  * `0` → Normal transaction
  * `1` → Fraudulent transaction

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Collection

The dataset is loaded using Pandas.

```python
import pandas as pd
df = pd.read_csv("creditcard.csv")
```

---

### 2️⃣ Data Exploration (EDA)

Basic dataset analysis:

* Dataset shape
* Statistical summary
* Class distribution

```python
df.head()
df.describe()
df['Class'].value_counts()
```

---

### 3️⃣ Handling Imbalanced Dataset

Since the dataset is highly imbalanced, **undersampling** is used to balance the classes.

```python
fraud = df[df.Class == 1]
normal = df[df.Class == 0]

normal_sample = normal.sample(len(fraud))

new_dataset = pd.concat([fraud, normal_sample], axis=0)
```

---

### 4️⃣ Feature and Target Separation

```python
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

---

### 5️⃣ Train-Test Split

The dataset is split into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

---

### 6️⃣ Model Training

Logistic Regression is used as the classification algorithm.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

---

### 7️⃣ Model Evaluation

Training Accuracy:

```python
from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
```

Testing Accuracy:

```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

---

## 📈 Model Performance

The model is evaluated using:

* Accuracy Score
* Training Accuracy
* Test Accuracy

Future improvements can include:

* Precision
* Recall
* F1 Score
* ROC-AUC Curve

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection
│
├── creditcard.csv
├── fraud_detection.ipynb
├── README.md
```

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to detect fraudulent transactions in financial systems. By handling imbalanced datasets and applying classification algorithms, we can build models that help prevent financial fraud.

---

## 👨‍💻 Author

**Saurav Nigam**
Machine Learning & Data Science Enthusiast

