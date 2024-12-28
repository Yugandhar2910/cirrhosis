"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st

def app():
    """This function create the home page"""
    
    # Add title to the home page
    st.title("Cirrhosis Detection System")

    # Add image to the home page
    st.image("./images/home.png")

    # Add brief describtion of your web app
    st.markdown(
    """<p style="font-size:20px;">
            Random Forest, a machine learning algorithm, can detect cirrhosis by analyzing various patient factors like liver enzymes, bilirubin levels, age, and alcohol consumption. It constructs numerous decision trees, each considering different features, and combines their outputs to determine the likelihood of cirrhosis. By training on datasets with known cirrhosis cases, Random Forest learns patterns and creates a predictive model. During testing, it assesses new data, evaluating the presence of cirrhosis based on the learned patterns, providing an efficient and accurate method for detection.
        </p>
    """, unsafe_allow_html=True)