import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.regex_extractor import extract_dimensions
from src.value_extractor import extract_value_from_text
from src.preprocess import MAX_LEN
import json
import os

def load_and_prepare_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer="adam", loss="binary_crossentropy")  # optional to suppress warnings
    return model

def extract_and_predict(model_path, tokenizer, mlb, description):
    model = load_and_prepare_model(model_path)

    seq = tokenizer.texts_to_sequences([description])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)[0]
    labels = mlb.inverse_transform(np.array([pred > 0.5]))[0]

    target_specs = ["Dimensions", "Standard", "Material", "Class Rating"]
    result = {label: None for label in target_specs}

    for label in target_specs:
        if label == "Dimensions":
            result[label] = extract_dimensions(description)
        else:
            result[label] = extract_value_from_text(label, description)

    return result

def predict_specs(model_path, tokenizer, mlb, input_file):
    model = load_and_prepare_model(model_path)

    with open(input_file) as f:
        test_data = json.load(f)

    descriptions = [item["input"] for item in test_data]
    actual_specs = [list(item["output"].keys()) for item in test_data]

    seq = tokenizer.texts_to_sequences(descriptions)
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    preds = model.predict(padded, verbose=0)

    predictions = []
    for i, pred in enumerate(preds):
        predicted_labels = mlb.inverse_transform(np.array([pred > 0.5]))[0]
        result = {label: None for label in ["Dimensions", "Standard", "Material", "Class Rating"]}
        result["Dimensions"] = extract_dimensions(descriptions[i])
        for label in predicted_labels:
            if label in result:
                result[label] = "Detected"

        predictions.append({
            "description": descriptions[i],
            "predicted_specifications": [k for k, v in result.items() if v],
            "actual_specifications": actual_specs[i]
        })

    os.makedirs("results", exist_ok=True)
    with open("results/output_predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    return predictions
