import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart_disease_uci.csv')
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(['id', 'num', 'dataset'], axis=1)

for col in ['fbs', 'exang']:
    df[col] = df[col].astype(str)

num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].fillna(df[col].mode()[0])

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

df = df.dropna()
X = df.drop('target', axis=1).astype(float)
y = df['target'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC()
model.fit(X_train, y_train)

st.title("❤️ Heart Disease Prediction")
st.write("Fill in the patient details and click Predict.")

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"])
ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])

if st.button("🔍 Predict"):
    input_dict = {
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalch': thalch, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([input_dict])
    for col in cat_cols:
        if col in input_df.columns:
            le = le_dict[col]
            val = str(input_df[col].iloc[0])
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0
    input_df = input_df.astype(float)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected! Please consult a doctor.")
    else:
        st.success("✅ No Heart Disease Detected!")
