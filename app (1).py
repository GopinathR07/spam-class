import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Spam Message Detection")

message = st.text_area("Enter your message")

if st.button("Predict"):

    msg = vectorizer.transform([message])

    prob = model.predict_proba(msg)[0][1]

    if prob > 0.5:
        st.error(f"Spam Message 🚨 (Confidence {prob:.2f})")
    else:
        st.success(f"Not Spam ✅ (Confidence {prob:.2f})")
