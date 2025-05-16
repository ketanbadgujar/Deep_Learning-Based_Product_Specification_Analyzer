from src.train import train_model
from src.predict import extract_and_predict, predict_specs
from src.evaluate import evaluate

def run_user_interface():
    tokenizer, mlb = train_model()

    predictions = predict_specs("models/spec_extractor.keras", tokenizer, mlb, "data/test_input.json")
    evaluate(predictions, mlb)

    print("\n🧠 Product Specification Extraction")
    print("Paste a product description like one from test_input.json.")
    print("Type 'exit' to quit.\n")

    while True:
        description = input("📝 Enter full product description:\n> ").strip()
        if description.lower() == 'exit':
            break

        result = extract_and_predict("models/spec_extractor.keras", tokenizer, mlb, description)

        print("\n🔍 Extracted Specifications:")
        for key, val in result.items():
            print(f"  {key}: {val if val else 'Not Found'}")

if __name__ == "__main__":
    run_user_interface()
