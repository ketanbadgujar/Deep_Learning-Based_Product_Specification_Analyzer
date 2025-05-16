from src.preprocess import load_data, preprocess_data, load_classes_from_all_sources
from src.model import create_model
import os

def train_model():
    all_classes = load_classes_from_all_sources("data/training_data.json", "data/test_input.json")
    texts, labels, _ = load_data("data/training_data.json")
    X, y, tokenizer, mlb = preprocess_data(texts, labels, all_classes)

    model = create_model(len(tokenizer.word_index) + 1, len(mlb.classes_))
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    os.makedirs("models", exist_ok=True)
    model.save("models/spec_extractor.keras")

    return tokenizer, mlb
