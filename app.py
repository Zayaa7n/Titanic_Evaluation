import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model + features
model = pickle.load(open("titanic_model.pkl", "rb"))
feature_cols = pickle.load(open("features.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details below:")

# Sidebar inputs
with st.sidebar:
    st.header("Passenger Info")

    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Age = st.slider("Age", 1, 80, 25)
    SibSp = st.slider("Siblings/Spouses", 0, 5, 0)
    Parch = st.slider("Parents/Children", 0, 5, 0)
    Fare = st.slider("Fare", 0, 500, 50)

    Sex = st.selectbox("Sex", ["male", "female"])
    Embarked = st.selectbox("Embarked", ["C", "Q", "S"])


# Convert to dataframe
input_data = pd.DataFrame({
    "Pclass":[Pclass],
    "Age":[Age],
    "SibSp":[SibSp],
    "Parch":[Parch],
    "Fare":[Fare],
    "Sex":[Sex],
    "Embarked":[Embarked]
})

# Same preprocessing as training
input_data = pd.get_dummies(input_data, columns=["Sex","Embarked"], drop_first=True)

# Add missing columns (important!)
for col in feature_cols:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[feature_cols]


# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {prob:.2f})")
