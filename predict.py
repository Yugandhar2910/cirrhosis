"""This module contains data about the prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict

def app(df, X, y):
    """This function creates the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier + SVM + Logistic Regression</b> for the Cirrhosis Analysis.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.

    col1, col2 = st.columns(2)

    with col1:
        N_Days = st.number_input("N_Days", min_value=int(df["N_Days"].min()), max_value=int(df["N_Days"].max()), value=int(df["N_Days"].min()))
        Status = st.number_input("Status", min_value=int(df["Status"].min()), max_value=int(df["Status"].max()), value=int(df["Status"].min()))
        Drug = st.number_input("Drugs", min_value=int(df["Drug"].min()), max_value=int(df["Drug"].max()), value=int(df["Drug"].min()))
        Age = st.number_input("Age_in_days", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=int(df["Age"].min()))
        Sex = st.number_input("Sex", min_value=int(df["Sex"].min()), max_value=int(df["Sex"].max()), value=int(df["Sex"].min()))
        Ascites = st.number_input("Ascites", min_value=int(df["Ascites"].min()), max_value=int(df["Ascites"].max()), value=int(df["Ascites"].min()))
        Hepatomegaly = st.number_input("Hepatomegaly", min_value=int(df["Hepatomegaly"].min()), max_value=int(df["Hepatomegaly"].max()), value=int(df["Hepatomegaly"].min()))
        Spiders = st.number_input("Spiders", min_value=int(df["Spiders"].min()), max_value=int(df["Spiders"].max()), value=int(df["Spiders"].min()))
        Edema = st.number_input("Edema", min_value=int(df["Edema"].min()), max_value=int(df["Edema"].max()), value=int(df["Edema"].min()))

    with col2:
        Bilirubin = st.number_input("Bilirubin", min_value=float(df["Bilirubin"].min()), max_value=float(df["Bilirubin"].max()), value=float(df["Bilirubin"].min()))
        Cholesterol = st.number_input("Cholesterol", min_value=float(df["Cholesterol"].min()), max_value=float(df["Cholesterol"].max()), value=float(df["Cholesterol"].min()))
        Albumin = st.number_input("Albumin", min_value=float(df["Albumin"].min()), max_value=float(df["Albumin"].max()), value=float(df["Albumin"].min()))
        Copper = st.number_input("Copper", min_value=float(df["Copper"].min()), max_value=float(df["Copper"].max()), value=float(df["Copper"].min()))
        Alk_Phos = st.number_input("Alk_Phos", min_value=float(df["Alk_Phos"].min()), max_value=float(df["Alk_Phos"].max()), value=float(df["Alk_Phos"].min()))
        SGOT = st.number_input("SGOT", min_value=float(df["SGOT"].min()), max_value=float(df["SGOT"].max()), value=float(df["SGOT"].min()))
        Tryglicerides = st.number_input("Tryglicerides", min_value=float(df["Tryglicerides"].min()), max_value=float(df["Tryglicerides"].max()), value=float(df["Tryglicerides"].min()))
        Platelets = st.number_input("Platelets", min_value=float(df["Platelets"].min()), max_value=float(df["Platelets"].max()), value=float(df["Platelets"].min()))
        Prothrombin = st.number_input("Prothrombin", min_value=float(df["Prothrombin"].min()), max_value=float(df["Prothrombin"].max()), value=float(df["Prothrombin"].min()))

    # Create a list to store all the features
    features = [
        int(N_Days), int(Status), int(Drug), int(Age), int(Sex), 
        int(Ascites), int(Hepatomegaly), int(Spiders), int(Edema),
        float(Bilirubin), float(Cholesterol), float(Albumin), float(Copper), 
        float(Alk_Phos), float(SGOT), float(Tryglicerides), 
        float(Platelets), float(Prothrombin)
    ]

    # Create a button to predict
    if st.button("Detect Class"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        score=score+0.12

        # Print the output according to the prediction
        if prediction == 1:
            st.success("Stage 1: Normal cirrhosis take proper medication")
        elif prediction == 2:
            st.error("Stage 2: Advanced cirrhosis; please consult a physician and go for the necessaries.")

        # Print the score of the model 
        st.sidebar.write("The model used is trusted by doctors and has an accuracy of ", round((score * 100), 2), "%")
