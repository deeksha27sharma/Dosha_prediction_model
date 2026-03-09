"""
Full Ayurveda Prediction Pipeline
---------------------------------

This pipeline predicts:

1. Dosha
2. Dosha probabilities
3. Disease
4. Treatment recommendations:
   - Therapy
   - Medicine
   - Diet
   - Exercise

Author: Diksha Sharma
"""

import pickle
import pandas as pd


# Load models
dosha_model = pickle.load(open("model/best_dosha_model.pkl", "rb"))
dosha_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

disease_model = pickle.load(open("model/disease_model.pkl", "rb"))
disease_encoder = pickle.load(open("model/disease_label_encoder.pkl", "rb"))


# Load dataset for treatment lookup
dataset = pd.read_csv("Dataset/Ayurvedic_ML_Dataset_3000_Records.csv")
dataset = dataset.fillna("Unknown")


def predict_ayurveda(input_dict):

    input_df = pd.DataFrame([input_dict])

    # -----------------------
    # DOSHA PREDICTION
    # -----------------------

    dosha_encoded = dosha_model.predict(input_df)
    predicted_dosha = dosha_encoder.inverse_transform(dosha_encoded)[0]

    dosha_probs = dosha_model.predict_proba(input_df)[0]

    dosha_probabilities = {
        dosha: float(prob)
        for dosha, prob in zip(dosha_encoder.classes_, dosha_probs)
    }

    # -----------------------
    # DISEASE PREDICTION
    # -----------------------

    input_df["Dosha"] = predicted_dosha

    disease_encoded = disease_model.predict(input_df)
    predicted_disease = disease_encoder.inverse_transform(disease_encoded)[0]

    # -----------------------
    # TREATMENT LOOKUP
    # -----------------------

    filtered = dataset[
        (dataset["Disease"] == predicted_disease) &
        (dataset["Dosha"] == predicted_dosha)
    ]

    if filtered.empty:
        filtered = dataset[dataset["Disease"] == predicted_disease]

    row = filtered.iloc[0]

    treatment = {
        "therapy": row["Therapy"],
        "medicine": row["Medicines"],
        "diet": row["Diet Plan"],
        "exercise": row["Exercise"]
    }

    # -----------------------
    # FINAL OUTPUT
    # -----------------------

    result = {

        "predicted_dosha": predicted_dosha,

        "dosha_probabilities": dosha_probabilities,

        "predicted_disease": predicted_disease,

        "treatment": treatment

    }

    return result


# Example usage
if __name__ == "__main__":

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

    output = predict_ayurveda(sample)

    print(output)

    sample1 = {

        "Age": 25,
        "Gender": "Female",
        "Prakriti": "Vata",
        "Symptoms": "inflammation, burning sensation, loose, burning stools",
        "Stress Level": "High",
        "Sleep Pattern": "Insomnia",
        "Diet Type": "Vegetarian",
        "Season": "Winter",
        "Climate": "Cold"
    }

    output1 = predict_ayurveda(sample1)

    print(output1)

    sample2 = {

        "Age": 49,
        "Gender": "Female",
        "Prakriti": "Kapha",
        "Symptoms": "sweet taste in mouth, lethargy, sweet-smelling urine",
        "Stress Level": "High",
        "Sleep Pattern": "Insomnia",
        "Diet Type": "Vegetarian",
        "Season": "Varsha",
        "Climate": "Cold"
    }

    output2 = predict_ayurveda(sample2)

    print(output2)

    sample3 = {

        "Age": 22,
        "Gender": "Female",
        "Prakriti": "kapha",
        "Symptoms": "cold,headache",
        "Stress Level": "High",
        "Sleep Pattern": "normal",
        "Diet Type": "Vegetarian",
        "Season": "Winter",
        "Climate": "Cold"
    }

    output3 = predict_ayurveda(sample3)

    print(output3)