 # Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""
    # Load the Cirrhosis dataset into DataFrame.
    df = pd.read_csv(r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\cirrhosis.csv")
    
   # Perform feature and target split
    X = df[["N_Days", "Status", "Drug", "Age", "Sex", "Ascites", 
         "Hepatomegaly", "Spiders", "Edema", "Bilirubin", 
         "Cholesterol", "Albumin", "Copper", "Alk_Phos", 
         "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]]
    y = df['Stage'].astype(int)


    return df, X, y

@st.cache_data()
def train_model(X, y):
    """This function trains the hybrid model (RF + SVM + Logistic Regression)"""
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define the base models
    base_models = [
        ('random_forest', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),  # SVM with probability=True for stacking
        ('logistic_regression', LogisticRegression(random_state=42, solver='liblinear'))
    ]

    # Use Logistic Regression as the meta-model
    meta_model = LogisticRegression(random_state=42, solver='liblinear')

    # Create a Stacking Classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # Fit the stacking model
    stacking_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    score = stacking_model.score(X_test, y_test)

    return stacking_model, score

def predict(X, y, features):
    """This function makes predictions using the trained hybrid model"""
    # Get model and model score from the training function
    model, score = train_model(X, y)
    
    # Predict the stage (assuming the model predicts 1 or 2)
    predicted_stage = model.predict(np.array(features).reshape(1, -1))[0]
    
    return predicted_stage, score