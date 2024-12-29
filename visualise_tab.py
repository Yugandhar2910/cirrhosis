"""This module contains data about the visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import streamlit as st

# Import necessary functions from web_functions
from web_functions import train_model

def app(df, X, y):
    """This function creates the visualisation page"""

    # Remove the warnings
    warnings.filterwarnings('ignore')

    # Set the page title
    st.title("Cirrhosis Analysis")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        # Create a heatmap plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.iloc[:, 1:].corr(), annot=True, ax=ax)  # Create heatmap using seaborn and pass axis

        # Adjust the limits to prevent top/bottom cutoff in seaborn heatmap
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        # Display the heatmap in the Streamlit app
        st.pyplot(fig)  # Use st.pyplot() to display the figure
