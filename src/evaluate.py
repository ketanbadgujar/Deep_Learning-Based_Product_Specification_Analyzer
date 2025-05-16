from sklearn.metrics import classification_report
import numpy as np

def evaluate(predictions, mlb):
    y_true, y_pred = [], []

    for item in predictions:
        actual = mlb.transform([item["actual_specifications"]])[0]
        predicted = mlb.transform([item["predicted_specifications"]])[0]
        y_true.append(actual)
        y_pred.append(predicted)

    print("\n📊 Evaluation Report:")
    print(classification_report(
        np.array(y_true),
        np.array(y_pred),
        target_names=mlb.classes_,
        zero_division=0
    ))
