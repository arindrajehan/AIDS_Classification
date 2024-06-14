import streamlit as st
import numpy as np
import pandas as pd
import joblib

from eda import eda_page
from prediction import model_page

#Load data
data = pd.read_csv("AIDS_Classification_5000.csv")

st.header('Milestone 2')
st.write("""
Created by Arindra Jehan - HCK015 """)

st.write("This program is made to predict whether the patient is healthy or infected with AIDS, based on `AIDS_Classification` database")
st.write("Dataset `AIDS_Classification`")
data

def main():
    # Define menu options
    menu_options = ["Data Analysis", "Model Prediction"]

    # Create sidebar menu
    selected_option = st.sidebar.radio("Menu", menu_options)

    # Display selected page
    if selected_option == "Data Analysis":
        eda_page()
    elif selected_option == "Model Prediction":
        model_page()


if __name__ == "__main__":
    main()