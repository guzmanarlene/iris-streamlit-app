# app.py
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements below:")

# Input fields
sepal_length = st.text_input("Sepal Length (cm)", "5.1")
sepal_width = st.text_input("Sepal Width (cm)", "3.5")
petal_length = st.text_input("Petal Length (cm)", "1.4")
petal_width = st.text_input("Petal Width (cm)", "0.2")

# Predict
if st.button("Predict Flower"):
    try:
        input_data = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]
        prediction = model.predict(input_data)
        flower = target_names[prediction[0]]
        st.success(f"Predicted Iris species: **{flower.capitalize()}** 🌼")
    except:
        st.error("Please enter valid numbers.")
