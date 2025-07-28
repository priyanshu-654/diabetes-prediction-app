🩺 Diabetes Prediction App

A Machine Learning web application that predicts the likelihood of diabetes based on user-input medical data. This app uses KNN, Decision Tree, and MLP Neural Network models, and is built with Streamlit for an interactive UI experience.

🔍 Overview

This project helps users determine whether a person is diabetic or not by inputting standard health parameters. It's ideal for learning applied machine learning, working with medical datasets, and deploying real-world data apps.

| Component        | Description                             |
| ---------------- | --------------------------------------- |
| 💻 Frontend      | Streamlit (Python-based Web UI)         |
| 🤖 ML Models     | KNN, Decision Tree, MLP Classifier      |
| 🧠 Preprocessing | StandardScaler from sklearn             |
| 📊 Dataset       | Pima Indians Diabetes Dataset (Kaggle)  |
| 📁 Tools         | Pandas, NumPy, Matplotlib, Scikit-learn |


📈 Features

✅ Predicts diabetes using 3 different models
✅ User inputs via interactive form
✅ Model selection: KNN / Decision Tree / MLP
✅ Real-time results displayed
✅ Built-in data scaling for MLP accuracy
✅ Clean and responsive UI with Streamlit

🧪 Input Parameters

The app takes the following medical attributes:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age

📦 diabetes-prediction-app/
├── app.py                  # Main Streamlit frontend
├── diabetes_model.py       # ML logic and model training
├── diabetes.csv            # Pima Indians dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

Install Dependencies:
pip install -r requirements.txt

Run the App:
streamlit run app.py









App Predicted Output:
<img width="647" height="841" alt="image" src="https://github.com/user-attachments/assets/864d8ea2-4897-4a8c-bc80-5c8af14f45eb" />

🙋‍♂️ Author
Priyanshu Pandey



