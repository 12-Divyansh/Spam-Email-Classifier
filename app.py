
import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Classifier")

message = st.text_area("Enter your email or SMS message")

if st.button("Predict"):
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not Spam")
