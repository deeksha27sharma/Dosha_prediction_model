"""
Disease Prediction Model using Ayurveda Dataset
Predicts disease using Dosha and lifestyle features
"""

import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Dataset/Ayurvedic_ML_Dataset_3000_Records.csv")

# Features and target
features = [
    "Age",
    "Gender",
    "Prakriti",
    "Symptoms",
    "Stress Level",
    "Sleep Pattern",
    "Diet Type",
    "Season",
    "Climate",
    "Dosha"
]

target = "Disease"

X = df[features].copy()
y = df[target].copy()

# Handle missing values
X["Age"] = X["Age"].fillna(X["Age"].median())

for col in X.columns:
    if col != "Age":
        X[col] = X[col].fillna("Unknown")

y = y.fillna("Unknown")

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# Preprocessing
categorical_cols = [
    "Gender",
    "Prakriti",
    "Stress Level",
    "Sleep Pattern",
    "Diet Type",
    "Season",
    "Climate",
    "Dosha"
]

text_col = "Symptoms"
numeric_cols = ["Age"]

preprocessor = ColumnTransformer([
    ("txt", TfidfVectorizer(max_features=300), text_col),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

# GridSearch
param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10, 20]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test)

print("Disease model accuracy:", accuracy)

# Save model
pickle.dump(best_model, open("model/disease_model.pkl", "wb"))
pickle.dump(label_encoder, open("model/disease_label_encoder.pkl", "wb"))

print("Disease model saved.")