import json
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50

def load_classes_from_all_sources(train_path, test_path):
    with open(train_path) as f1:
        train_data = json.load(f1)["training_data"]
    with open(test_path) as f2:
        test_data = json.load(f2)

    train_classes = {item["product_specification"] for item in train_data}
    test_classes = set()

    for item in test_data:
        test_classes.update(item["output"].keys())

    return sorted(list(train_classes.union(test_classes)))

def load_data(file_path):
    with open(file_path) as f:
        raw_data = json.load(f)["training_data"]

    texts, labels = [], []
    for item in raw_data:
        spec = item["product_specification"]
        for val in item["specification_value"]:
            texts.append(val)
            labels.append([spec])

    return texts, labels, []

def preprocess_data(texts, labels, all_classes):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=MAX_LEN)

    mlb = MultiLabelBinarizer(classes=all_classes)
    y = mlb.fit_transform(labels)

    return X, y, tokenizer, mlb
