import streamlit as st
from joblib import load

# Load trained model and CountVectorizer
lr = load('logistic_regression_model.joblib')
cv = load('countvectorizer.joblib')

# App title
st.title("📨 Spam Message Detector")

# Input field
user_input = st.text_input("✏️ Enter a sentence to check if it's Spam or Ham:")

# Prediction button
if st.button('🔍 Predict Spam/Ham'):
    if user_input.strip():  # Check if input is not empty
        Snew = cv.transform([user_input])  # Wrap in list
        prediction = lr.predict(Snew)[0]  # Get the result

        # Display the result nicely
        if prediction == 'spam':
            st.error("🚫 This message is predicted to be **SPAM**.")
        else:
            st.success("✅ This message is predicted to be **HAM** (Not Spam).")
    else:
        st.warning("⚠️ Please enter a sentence first.")
