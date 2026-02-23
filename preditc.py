"""
Ayurveda Dosha Prediction System
-------------------------------------------

File: predict.py

Description:
This script loads the trained Machine Learning model and predicts the Ayurvedic Dosha
(Vata, Pitta, Kapha) based on user input features.

Features:
- Loads optimized model trained using GridSearchCV
- Predicts Dosha based on input features
- Displays confidence score for each Dosha
- Returns final predicted Dosha with highest confidence

Required Files:
- model/best_dosha_model.pkl
- model/label_encoder.pkl

Author: Diksha Sharma
Project: Dosha Prediction System
"""

import pickle
import pandas as pd

# load model
model = pickle.load(open("model/best_dosha_model.pkl", "rb"))

# load label encoder
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

def predict_dosha_with_confidence(input_dict):

    input_df = pd.DataFrame([input_dict])

    # prediction
    prediction_encoded = model.predict(input_df)
    predicted_dosha = label_encoder.inverse_transform(prediction_encoded)[0]

    # probabilities
    probabilities = model.predict_proba(input_df)[0]
    class_names = label_encoder.classes_

    # convert to clean float %
    confidence = {
        dosha: float(round(prob * 100, 2))
        for dosha, prob in zip(class_names, probabilities)
    }

    # find highest confidence dosha
    final_dosha = max(confidence, key=confidence.get)

    # Output section
    print(f"'predicted_dosha': '{predicted_dosha}'")

    print("\nConfidence levels:")
    for dosha, prob in confidence.items():
        print(f"{dosha}: {prob:.2f}%")

    print(f"\nFinal Output: {final_dosha}")

    return final_dosha

#Example
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

predict_dosha_with_confidence(sample)
