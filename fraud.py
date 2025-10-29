import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Gamified Fraud Detection", layout="wide")

st.title("💰 Gamified Credit Card Fraud Detection")

st.markdown("""
This application simulates a transaction and uses a trained Logistic Regression model to predict if it's fraudulent or legitimate.
Enter the transaction details below to test the model!
""")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('logistic_regression_model.pkl')
        scaler = joblib.load('standard_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'logistic_regression_model.pkl' and 'standard_scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

if model and scaler:
    st.header("Enter Transaction Details")

    # Using columns to organize the input fields
    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("Time (seconds elapsed from first transaction)", value=0.0, format="%f")
        amount = st.number_input("Transaction Amount", value=0.0, format="%f")

    st.subheader("PCA Transformed Features (V1 - V28)")
    # Organize V features using columns
    v_columns = [f"V{i}" for i in range(1, 29)]
    v_values = {}
    # Create columns dynamically
    num_cols = 4
    cols = st.columns(num_cols)
    for i, v_col in enumerate(v_columns):
        with cols[i % num_cols]:
            v_values[v_col] = st.number_input(v_col, value=0.0, format="%f", key=v_col)


    if st.button("Simulate Transaction and Detect Fraud"):
        transaction_data = {'Time': time}
        transaction_data.update(v_values)
        transaction_data['Amount'] = amount # Add amount here

        # Ensure the DataFrame has the correct column order
        # The original columns based on the dataset
        original_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


        try:
            input_df = pd.DataFrame([transaction_data], columns=original_columns)

            # Display the input data (optional)
            st.write("Input Transaction Data:")
            st.write(input_df)

            # Scale the input data
            input_scaled = scaler.transform(input_df)

            # Make a prediction
            prediction = model.predict(input_scaled)

            st.subheader("Detection Result:")
            # Display the prediction result in a gamified way
            if prediction[0] == 1:
                st.error("🚨 Fraudulent Transaction Detected! 🚨")
                st.balloons()
            else:
                st.success("✅ Legitimate Transaction. You're Safe! ✅")
                st.snow()

        except ValueError as e:
            st.error(f"Error processing input data: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
