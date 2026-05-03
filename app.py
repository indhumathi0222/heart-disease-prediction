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

# Convert all columns to string first, then handle
for col in df.columns:
    if col == 'target':
        continue
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        pass

# Fix missing values
for col in df.columns:
    if col == 'target':
        continue
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col] = df[col].astype(str)
        df[col].fillna(df[col].mode()[0], inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

df = df.dropna()

# Train model
X = df.drop('target', axis=1).astype(float)
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
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
cp = st.selectbox("Chest Pain Type", [0,1,2,3],
    format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0],
    format_func=lambda x: "Yes" if x==1 else "No")
restecg = st.selectbox("Resting ECG", [0,1,2],
    format_func=lambda x: ["Normal","LV Hypertrophy","ST-T Abnormality"][x])
thalch = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [1, 0],
    format_func=lambda x: "Yes" if x==1 else "No")
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0,1,2],
    format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
ca = st.slider("Number of Vessels", 0, 3, 0)
thal = st.selectbox("Thal", [0,1,2],
    format_func=lambda x: ["Normal","Fixed Defect","Reversable Defect"][x])

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalch, exang, oldpeak,
                                  slope, ca, thal]],
                               columns=X.columns)
    input_data = input_data.astype(float)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease!")
