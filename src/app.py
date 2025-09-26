import streamlit as st
import numpy as np
import pandas as pd
from pickle import load
import json

st.title('Medical Charge Prediction')

# Charge of the model and the scaler
try:
    model = load(open('src/final_model.pkl', 'rb'))
    scaler = load(open('src/scaler_without_outliers.pkl', 'rb'))
except FileNotFoundError:
    model = load(open('final_model.pkl', 'rb'))
    scaler = load(open('scaler_without_outliers.pkl', 'rb'))


# Factorization
with open('./src/factorize_data/factorized_region.json') as f:
    region_dict = json.load(f)
with open('./src/factorize_data/factorized_sex.json') as f:
    sex_dict = json.load(f)
with open('./src/factorize_data/factorized_smoker.json') as f:
    smoker_dict = json.load(f)

# Inputs of the user
age = st.slider("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", list(sex_dict.keys()))
bmi = st.slider("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.slider("Children", min_value=0, max_value=12, step=1)
smoker = st.selectbox("Smoker", list(smoker_dict.keys()))
region = st.selectbox("Region", list(region_dict.keys()))

# Encoding
sex_code = sex_dict[sex]
smoker_code = smoker_dict[smoker]
region_code = region_dict[region]
has_children = 1 if children > 0 else 0

# Calculate has_children variable
has_children = 1 if children > 0 else 0

# Create the input array with all variables
row = np.array([[age, sex_code, bmi, children, smoker_code, region_code, has_children]])

if st.button('Predict'):
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    st.success(f"Predicted medical charge: {prediction:.2f}$")