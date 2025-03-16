import streamlit as st
import pickle
import numpy as np

# Inject custom CSS styles
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Apply font to entire app */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f4f4f4, #d4edda);
        color: #333;
    }

    /* Style for the title */
    .title {
        color: #2E86C1;
        font-size: 36px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Card container */
    .card {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 60%;
        margin: auto;
    }

    /* Input field styling */
    .stNumberInput>div>div>input {
        border: 2px solid #2E86C1;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(to right, #2E86C1, #1F618D);
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1F618D, #2E86C1);
        transform: scale(1.05);
    }

    /* Success & Error Messages */
    .stAlert[data-testid="stAlert-success"] {
        background-color: #D4EDDA;
        color: #155724;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
    }

    .stAlert[data-testid="stAlert-error"] {
        background-color: #F8D7DA;
        color: #721C24;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
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
st.markdown('<div class="card">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)
