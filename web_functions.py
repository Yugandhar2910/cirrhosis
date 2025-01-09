# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib  # To save and load the model and scaler

@st.cache_data()
def load_data():
    """Load and preprocess the Cirrhosis dataset."""
    df = pd.read_csv(r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\balanced_cirrhosis.csv")
    
    # Perform feature and target split
    X = df[[
        "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", 
        "Hepatomegaly", "Spiders", "Edema", "Bilirubin", 
        "Cholesterol", "Albumin", "Copper", "Alk_Phos", 
        "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
    ]]
    y = df['Stage'].astype(int)

    return df, X, y

@st.cache_data()
def train_model(X, y):
    """Train the hybrid model (SVM + RF + Logistic Regression)."""
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Define the base models
    base_models = [
        ('random_forest', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]

    # Use Logistic Regression as the meta-model
    meta_model = LogisticRegression(random_state=42, solver='liblinear')

    # Create a Stacking Classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # Fit the stacking model
    stacking_model.fit(X_train, y_train)

    # Save the trained model and scaler
    joblib.dump(stacking_model, "stacking_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Evaluate the model on the test set
    score = stacking_model.score(X_test, y_test)

    return stacking_model, score

def predict(X, y, features):
    """Make predictions using the trained hybrid model."""
    try:
        # Load the trained model and scaler
        stacking_model = joblib.load("stacking_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        # Train the model if not found
        stacking_model, score = train_model(X, y)
    else:
        # Score the loaded model
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        score = stacking_model.score(X_test, y_test)

    # Scale the input features
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))

    # Predict the stage
    predicted_stage = stacking_model.predict(features_scaled)[0]

    return predicted_stage, score
