import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data from a CSV file
data = pd.read_csv('AIDS_Classification_5000.csv')

def eda_page():

    st.title("Exploratory Data Analysis")
    st.write('Data exploration is made to better understand the dataset')
    st.subheader("Distribution of Infected and Non-Infected Patient")

    # Make plot pie on infected patients
    infected_pie_chart_fig, ax = plt.subplots(figsize=(10, 8))

    # Set the background color of the plot to dark
    ax.patch.set_facecolor('#333333')

    # Create the pie chart
    ax.pie(data['infected'].value_counts(), labels=['Non Infected', 'Infected'], 
            explode=[0, 0.1], autopct='%.0f%%', colors=['#007bff', '#ffa07a'])

    # Display the plot using Streamlit
    st.pyplot(infected_pie_chart_fig)
    st.write("**Description**:")
    st.write('Based on the figure above, there is an imbalance between an Infected and Non-Infected Patient. The figure shows that there are 68% of Non-Infected patient and there are only 32% of Infected Patient')
    st.write()
    
    st.subheader("Patient Demographics Distribution")

    # Define the columns
    cols = ['race', 'gender']

    # Create a figure with 1x2 grid of subplots
    countplot_fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # adjusted figsize

    # Loop through the columns and create a countplot for each column
    for i, col in enumerate(cols):
        ax = axs[i]
        sns.countplot(data=data, x=col, hue=data['infected'], ax=ax, palette='pastel')

    # Display the plot using Streamlit
    st.pyplot(countplot_fig)
    st.write("**Description**:")
    st.write('Based on the figure above, we can see the Demographics of the Infected Patients :')
    st.write('* Race: Majority of patients are white, with fewer non-white patients.')
    st.write('* Gender: More male patients than female patients.')

    st.subheader("Patient Medical History Distribution")

    # Define the columns
    cols = ['hemo', 'homo', 'drugs', 'karnof', 'oprior', 'z30']

    # Create a figure with 3x2 grid of subplots
    countplot_fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

    # Loop through the columns and create a countplot for each column
    for i, col in enumerate(cols):
        ax = axs[i//2, i%2]
        sns.countplot(data=data, x=col, hue=data['infected'], ax=ax, palette='pastel')

    # Display the plot using Streamlit
    st.pyplot(countplot_fig)
    st.write("**Description**:")
    st.write('Based on the figure above, we can see the Medical History of the Infected Patients :')
    st.write('* Hemophilia: More patients without hemophilia than with hemophilia.')
    st.write('* Homosexual Activity: More patients with no history of homosexual activity than with such history.')
    st.write('* IV Drug Use: More patients with no history of IV drug use than with such history.')
    st.write('* Karnofsky Score: Most patients have a Karnofsky score of 100.')
    st.write('* Pre-175 Antiretroviral Therapy: Most patients did not undergo Non-ZDV antiretroviral therapy pre-175.')
    st.write('* ZDV Usage: More patients with ZDV in the 30 days prior to 175 than without.')

    st.subheader("Patient Treatment Distribution")

    # Define the columns
    cols = ['trt', 'treat', 'offtrt']

    # Create a figure with 3x1 grid of subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    # Loop through the columns and create a countplot for each column
    for i, col in enumerate(cols):
        sns.countplot(data=data, x=col, hue=data['infected'], ax=axs[i], palette='pastel')

    # Display boxplot using Streamlit
    st.pyplot(fig)
    st.write("**Description**:")
    st.write('Based on the figure above, we can see the Treatment of the Infected Patients :')
    st.write('* ZDV vs. ddI: More patients on ZDV only treatment than ddI only treatment, followed by ZDV + Zal treatment and ZDV + ddI treatment.')
    st.write('* Other Treatment: More patients with other treatments than with ZDV only treatment.')
    st.write('* Off-trt: Most patients do not show indication of off-trt before 96+/-5 weeks.')

    st.subheader("Patient Treatment History Distribution")

    # Define the columns
    cols = ['str2', 'strat']

    # Create a figure with 1x2 grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))  # adjusted figsize

    # Loop through the columns and create a countplot for each column
    for i, col in enumerate(cols):
        sns.countplot(data=data, x=col, hue=data['infected'], ax=axs[i], palette='pastel')

    # Display the plot using Streamlit
    st.pyplot(fig)
    st.write("**Description**:")
    st.write('Based on the figure above, we can see the Treatment History of the Infected Patients :')
    st.write('* Antiretroviral Experience: More patients with antiretroviral history than naive patients.')
    st.write('* Antiretroviral History Stratification: More patients with no or less than 52 weeks of antiretroviral history than those with more than 52 weeks.')

    st.subheader("Patient Symptom Distribution")

    # Define the columns
    cols = ['symptom']

    # Create a figure with 1x1 grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6)) 

    # Create a countplot for the column
    sns.countplot(data=data, x=cols[0], hue=data['infected'], ax=axs, palette='pastel')

    # Display the plot using Streamlit
    st.pyplot(fig)
    st.write("**Description**:")    
    st.write('Based on the figure above, we can see the Symptoms of the Infected Patients :')
    st.write('* Symptoms: More asymptomatic patients than symptomatic patients.')
    st.write()

    st.subheader("Check Data Outliers")

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))

    # Define the columns
    cols = ['time', 'age', 'wtkg', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']

    # Loop through the columns and create a boxplot for each column
    for i, col in enumerate(cols):
        axs[i//2, i%2].set_title(col)
        sns.boxplot(data=data, y=col, ax=axs[i//2, i%2])

    # Display the plot using Streamlit
    st.pyplot(fig)
    st.write("**Description**:")    
    st.write('Based on the figure above, there are many outliers in the data. This should be handled, because :')
    st.write('Outliers can have a significant impact on statistical analysis and machine learning models, as they can skew the results and lead to incorrect conclusions.')
    st.write()