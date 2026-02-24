"""
Ayurveda Dosha Prediction model
-------------------------------------------

File: train.py

Description:
This script trains and optimizes a Machine Learning model to predict Ayurvedic Dosha 
(Vata, Pitta, Kapha) using physiological, lifestyle, and symptom-based features.

Pipeline includes:
- Data preprocessing and missing value handling
- Feature encoding using TF-IDF and OneHotEncoder
- Train-test split with stratification
- Hyperparameter optimization using GridSearchCV
- RandomForest model training
- Best model selection and saving for deployment

Output Files:
- best_dosha_model.pkl → trained optimized model
- label_encoder.pkl → label encoder for decoding predictions
- grid_search.pkl → GridSearchCV object for analysis

Author: Diksha Sharma
Project: Dosha Prediction model
"""


import pandas as pd
df = pd.read_csv("Dataset/Ayurvedic_ML_Dataset_3000_Records.csv")


features = [
    "Age", "Gender", "Prakriti","Symptoms",
    "Stress Level", "Sleep Pattern",
    "Diet Type", "Season", "Climate"
]

target = "Dosha"
X = df[features].copy()
y = df[target].copy()

X.loc[:, "Age"] = X["Age"].fillna(X["Age"].median())

for col in X.columns:
    if col != "Age":
        X.loc[:, col] = X[col].fillna("Unknown")

y = y.fillna("Unknown")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

categorical_cols = [
    "Gender","Prakriti","Stress Level",
    "Sleep Pattern", "Diet Type",
    "Season", "Climate"
]

text_col = "Symptoms"
numeric_cols = ["Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("txt", TfidfVectorizer(max_features=300), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {

    "model__n_estimators": [100, 200, 300],

    "model__max_depth": [None, 10, 20],

    "model__min_samples_split": [2, 5],

    "model__min_samples_leaf": [1, 2]

}

grid_search = GridSearchCV(

    pipeline,

    param_grid,

    cv=5,                 # 5-fold cross validation

    scoring="accuracy",

    n_jobs=-1,            # use all CPU cores

    verbose=1

)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("Best CV Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test)

print("Best Accuracy:", accuracy)

import pickle

pickle.dump(best_model, open("model/best_dosha_model.pkl", "wb"))
pickle.dump(label_encoder, open("model/label_encoder.pkl", "wb"))
pickle.dump(grid_search, open("model/grid_search.pkl", "wb"))

print("Best model saved successfully")
