import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('heart_disease_uci.csv')
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(['id', 'num', 'dataset'], axis=1)

# Fix missing values correctly
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encoding
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'boolean':
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Convert all to float
df = df.astype(float)
df = df.dropna()

# Train model
X = df.drop('target', axis=1)
y = df['target'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC()
model.fit(X_train, y_train)

# App UI
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below:")

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", ["True", "False"])
restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
ca = st.slider("Number of Vessels", 0, 3, 0)
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalch, exang, oldpeak,
                                  slope, ca, thal]],
                               columns=X.columns)
    for col in le_dict:
        if col in input_data.columns:
            try:
                input_data[col] = le_dict[col].transform(
                    input_data[col].astype(str))
            except:
                input_data[col] = 0
    input_data = input_data.astype(float)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease!")
