import streamlit as st
import pickle

preprocess_func = pickle.load(open('preprocess_text.pkl','rb'))
bow = pickle.load(open('bow_vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_message = st.text_area("Enter the message")

if st.button('Predict'):

    preprocessed_message = preprocess_func(input_message)
    vectorized_input = bow.transform([preprocessed_message])
    result = model.predict(vectorized_input)[0]

    if result == 1:
        st.header("The Email is **SPAM**")
    else:
        st.header("The Email is **NOT SPAM**")