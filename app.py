import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load and train
df = pd.read_csv('heart_disease_uci.csv')
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(['id', 'num', 'dataset'], axis=1)

num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

le = LabelEncoder()
encode_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in encode_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC()
model.fit(X_train, y_train)

# App
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
    for col in encode_cols:
        input_data[col] = le.fit_transform(input_data[col].astype(str))
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease!")