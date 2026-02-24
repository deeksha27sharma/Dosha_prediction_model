# Dosha_prediction_model
# 🧠 Ayurveda Dosha Prediction System

A Machine Learning system that predicts a person’s Ayurvedic Dosha (Vata, Pitta, Kapha) using physiological, lifestyle, and symptom data. The model is optimized using GridSearchCV and provides confidence scores for each prediction.

This project demonstrates a complete end-to-end ML pipeline from preprocessing to deployment-ready inference.

# 📌 Features

✅ Predicts Dosha using Machine Learning

✅ Confidence score for each Dosha

✅ TF-IDF processing for symptom text

✅ One-Hot Encoding for categorical features

✅ Hyperparameter tuning using GridSearchCV

✅ Automatic best model selection

✅ Model saving and loading using Pickle

✅ Production-ready pipeline


# 🧬 Input Features

The model uses the following features:

1. Age

2. Gender

3. Prakriti

4. Symptoms

5. Stress Level

6. Sleep Pattern

7. Diet Type

8. Season

9. Climate




# 🤖 Machine Learning Pipeline

```
Dataset
│
├── Data Cleaning
│
├── Feature Encoding
│   ├── TF-IDF Vectorizer (Symptoms)
│   ├── OneHotEncoder (Categorical Features)
│   └── Numeric Features (Age)
│
├── Train-Test Split
│
├── Model Comparison
│   ├── Logistic Regression
│   ├── Decision Tree
│   ├── Random Forest
│   ├── Gradient Boosting
│   └── XGBoost
│
├── GridSearchCV Hyperparameter Optimization
│
├── Best Model Selection
│
├── Model Saving (Pickle)
│
└── Prediction with Confidence Scores
```

# 🔧 Technologies Used

* Python

* Pandas

* NumPy

* Scikit-Learn

* XGBoost

* GridSearchCV

* TF-IDF Vectorizer

* Pickle

# 📊 Model Optimization

Hyperparameter tuning was performed using GridSearchCV to find the best Random Forest model configuration.

## Parameter Grid
```
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
```
## Optimization Method

The following optimization techniques were used:

5-Fold Cross Validation

Parallel Processing using all CPU cores (n_jobs = -1)

Automated Best Model Selection based on accuracy
```

RandomForestClassifier
   ↓
GridSearchCV
   ↓
Cross Validation (5 folds)
   ↓
Best Hyperparameters Selected
   ↓
Best Model Saved (Pickle)
```
## Best Model Features

Optimized Random Forest model

Fully integrated preprocessing pipeline

TF-IDF feature vectorization for symptoms

OneHotEncoding for categorical features

Production-ready saved model

# 📈 Example Output
```
'predicted_dosha': 'Vata'

Confidence levels:
Kapha: 0.00%
Pitta: 0.00%
Vata: 100.00%

Final Output: Vata
```

# 📁 Project Structure
```
dosha-prediction/
│
├── dataset/
│   └── Ayurvedic_ML_Dataset_3000_Records.csv
│
├── model/
│   ├── best_dosha_model.pkl
│   └── label_encoder.pkl
│
├── train_model.py
├── predict.py
└── README.md
```
# 🚀 Installation
## Clone the repository:
```
git clone https://github.com/deeksha27sharma/dosha_model_prediction.git
cd dosha_model_prediction
```
## Install dependencies:
```
pip install pandas numpy scikit-learn xgboost
```

# ▶️ Usage

## Train the model
```
python train_model.py
```
## Make prediction
```
predict_dosha_with_confidence(sample_input)
```

# 🧪 Example Prediction Code
```
sample = {
    "Age": 25,
    "Gender": "Female",
    "Prakriti": "Vata",
    "Symptoms": "dry skin, anxiety, constipation",
    "Stress Level": "High",
    "Sleep Pattern": "Insomnia",
    "Diet Type": "Vegetarian",
    "Season": "Winter",
    "Climate": "Cold"
}
```
# 🏆 Project Highlights

* End-to-end ML pipeline implementation

* Hyperparameter optimized model

* Confidence score prediction

* Deployment-ready architecture

* Clean and modular code design

# 📊 Project Status

* Status: Complete
* Level: Advanced Machine Learning Project
* Deployment Ready: Yes

# 👩‍💻 Author

Diksha Sharma

BTech Computer Science Engineering

# ⭐ Future Improvements

* Streamlit Web App

* FastAPI Deployment

* Real-time prediction API

* Integration with healthcare applications
  

# 📁 Dataset

The dataset used in this project was created and provided by my project partner, Jagveer Singh Bedi.


