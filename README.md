# Deep Learning-Based Product Specification Analyzer  

This project is a deep learning application that automatically extracts **product specifications** (such as Dimensions, Standard, Material, and Class Rating) from free-form product descriptions. It combines **CNN + BiLSTM** models with **Regex-based enhancements** for higher accuracy.  

---

## ✨ Features  
- Extracts structured product specs from unstructured text.  
- Deep learning model using **TensorFlow & Keras**.  
- **Regex patterns** for attributes like Dimensions.  
- User-friendly prediction pipeline.  
- Flask API for deployment.  

---

## 📂 Project Structure  

├── data/
│   ├── training_data.json
│   ├── input_data.json
│   └── test_input.json
│
├── models/
│   └── product_spec_analyzer.h5
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── predict.py
│   ├── regex_patterns.py
│   └── evaluate.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
