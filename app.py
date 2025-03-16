import streamlit as st
import pickle
import numpy as np
import sklearn

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Background and text styles */
    body {
        background-color: #f4f4f4;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    
    /* Title Styling */
    .title {
        color: #2E86C1;
        font-size: 32px;
        text-align: center;
        font-weight: bold;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #1F618D;
        color: #FFF;
    }

    /* Input Field Styling */
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #2E86C1;
        padding: 8px;
    }
    
    /* Success Message */
    .stAlert[data-testid="stAlert-success"] {
        background-color: #D4EDDA;
        color: #155724;
        font-weight: bold;
    }

    /* Error Message */
    .stAlert[data-testid="stAlert-error"] {
        background-color: #F8D7DA;
        color: #721C24;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaling.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit UI
st.markdown('<h1 class="title">🩺 Diabetes Prediction App</h1>', unsafe_allow_html=True)
st.write("Fill in the details below and click **Predict Diabetes**.")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f", step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f", step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

# Button to make prediction
if st.button("Predict Diabetes"):
    try:
        # Create input array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

        # Scale the input
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Show result
        if prediction[0] == 1:
            st.error("⚠️ The model predicts **Diabetes**.")
        else:
            st.success("✅ The model predicts **No Diabetes**.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
