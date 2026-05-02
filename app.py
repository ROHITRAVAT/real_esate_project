import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model, encoder = pickle.load(open('model/model.pkl', 'rb'))

st.title("Real Estate Price Predictor")

# Inputs
area = st.number_input("Area (sq ft)", 500, 5000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
location = st.selectbox("Location", encoder.classes_)

# Encode location
location_encoded = encoder.transform([location])[0]

# Prediction
if st.button("Predict Price"):
    input_data = [[area, bedrooms, bathrooms, location_encoded]]
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹{prediction:,.2f}")

# Visualization
st.subheader("Data Visualization")

df = pd.read_csv('data/housing.csv')

fig, ax = plt.subplots()
ax.scatter(df['area'], df['price'])
ax.set_xlabel("Area")
ax.set_ylabel("Price")
ax.set_title("Area vs Price")

st.pyplot(fig)

#train the model, run with `python train.py` and then run the app with `python -m streamlit run app.py` in the terminal to see the visualization.