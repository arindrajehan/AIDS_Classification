import streamlit as st
import numpy as np
import pandas as pd
import joblib

def model_page():
    st.title("Model Prediction of AIDS Infection")
    st.write("The model predict whether the patient is infected or not infected with AIDS")
    st.sidebar.header('User Input Features')

    input_data = user_input()

    st.subheader('User Input')
    st.write(input_data)

    load_model = joblib.load("knn_best_model_pipeline.joblib")

    prediction = load_model.predict(input_data)

    if prediction == 1:
        prediction = 'The patient is infected with AIDS'
    else:
        prediction = 'The patient is not infected with AIDS'

    st.write('Based on user input, the model predicted: ')
    st.write(prediction)

def user_input(num_rows=1):
    time = st.sidebar.number_input('time', 124, 1231, 124)
    trt = st.sidebar.selectbox('trt', [0, 1, 2, 3])
    age = st.sidebar.number_input('age', 12, 62, 12)
    wtkg = st.sidebar.number_input('wtkg', 44, 142, 44)
    hemo = st.sidebar.selectbox('hemo', [0, 1])
    homo = st.sidebar.selectbox('homo', [0, 1])
    drugs = st.sidebar.selectbox('drugs', [0, 1])
    karnof = st.sidebar.number_input('karnof', 78, 100, 78)
    oprior = st.sidebar.selectbox('oprior', [0, 1])
    z30 = st.sidebar.selectbox('z30', [0, 1])
    preanti = st.sidebar.number_input('preanti', 0, 2351, 0)
    race = st.sidebar.selectbox('race', [0, 1])
    gender = st.sidebar.selectbox('gender', [0, 1])
    str2 = st.sidebar.selectbox('str2', [0, 1])
    strat = st.sidebar.selectbox('strat', [1, 2, 3])
    symptom = st.sidebar.selectbox('symptom', [0, 1])
    treat = st.sidebar.selectbox('treat', [0, 1])
    offtrt = st.sidebar.selectbox('offtrt', [0, 1])
    cd40 = st.sidebar.number_input('cd40', 115, 716, 115)
    cd420 = st.sidebar.number_input('cd420', 119, 1104, 119)
    cd80 = st.sidebar.number_input('cd80', 252, 4922, 252)
    cd820 = st.sidebar.number_input('cd820', 236, 3055, 236)
        
    data = {
        'time': time,
        'trt': trt,
        'age': age,
        'wtkg': wtkg,
        'hemo': hemo,
        'homo': homo,
        'drugs': drugs,
        'karnof': karnof,
        'oprior': oprior,
        'z30': z30,
        'preanti': preanti,
        'race': race,
        'gender': gender,
        'str2': str2,
        'strat': strat,
        'symptom': symptom,
        'treat': treat,
        'offtrt': offtrt,
        'cd40': cd40,
        'cd420': cd420,
        'cd80': cd80,
        'cd820': cd820
        }
    features = pd.DataFrame(data, index=[0])
    return features