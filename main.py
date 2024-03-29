import numpy as np
import streamlit as st
import pandas as pd
import joblib
import time

if 'showAlert' not in st.session_state:
    st.session_state.showAlert = False

if 'prediction' not in st.session_state:
    st.session_state.prediction = ''

model = joblib.load('model.pkl')

loaded_label_encoders = {}
for col in ['CAEC', 'CALC']:
    loaded_label_encoders[col] = joblib.load(f'encoders/{col}_encoder.pkl')

one_hot_encoder = joblib.load('encoders/one_hot_encoder.pkl')
std_scaler = joblib.load('std_scaler.pkl')

st.title('Multi-Class Prediction of Obesity Risk!')


Height = st.number_input('Height: ')
Weight = st.number_input('Weight: ')
Gender = st.radio('Select Gender', ['Male', 'Female'])
Age = st.slider('Patient Age: ', 17, 100)
family_history_with_overweight = st.selectbox('Family History With Overweight', ['yes', 'no'])
FAVC = st.selectbox('Frequency of consumption of high caloric food (FAVC)', ['yes', 'no'])
FCVC = st.number_input('Frequency of consumption of vegetables (FCVC): ')
NCP = st.number_input('Number of main meals per day (NCP): ')
CAEC = st.selectbox('Consumption of food between meals (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
SMOKE = st.selectbox('Are you a smoker', ['yes', 'no'])
CH2O = st.number_input('Consumption of water daily (CH2O): ')
SCC = st.selectbox('Calories consumption monitoring (SCC)', ['yes', 'no'])
FAF = st.number_input('Frequency of physical activity (FAF): ')
TUE = st.number_input('Time using technology devices (TUE): ')
CALC = st.selectbox('Caloric consumption monitoring (CALC)', ['no', 'Sometimes', 'Frequently'])
MTRANS = st.selectbox('Mode of transportation', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])


def predict():
    data = [family_history_with_overweight, FAVC, SMOKE, SCC, MTRANS]

    input_df = pd.DataFrame([data],
                            columns=['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS'])
    encoded_inputs = one_hot_encoder.transform(input_df)

    CAEC_encoded = loaded_label_encoders['CAEC'].transform([CAEC])
    CALC_encoded = loaded_label_encoders['CAEC'].transform([CALC])
    Gender_encoded = 0
    if Gender == 'Male': Gender_encoded = 1

    row = np.array([Gender_encoded, Age, Height, Weight, FCVC, NCP, CAEC_encoded[0], CH2O, FAF, TUE, CALC_encoded[0]])
    final_data = np.concatenate([row, encoded_inputs.values[0]])

    scaled_data = std_scaler.transform([final_data])
    prediction = model.predict(scaled_data)
    print('prediction', prediction[0])

    label_mapping = {'Overweight_Level_II': 0, 'Normal_Weight': 1, 'Insufficient_Weight': 2,
                     'Obesity_Type_III': 3, 'Obesity_Type_II': 4, 'Overweight_Level_I': 5, 'Obesity_Type_I': 6}

    def decode_predictions(prediction_, label_mapping_):
        for key, value in label_mapping_.items():
            if value == prediction_:
                return key

    decoded_prediction = decode_predictions(prediction[0], label_mapping)
    print(decoded_prediction)

    with st.spinner('Wait for it...'):
        time.sleep(3)
        st.session_state.showAlert = True
        st.session_state.prediction = decoded_prediction


st.button('Predict', on_click=predict)

if st.session_state.showAlert:
    prediction = st.session_state.prediction
    if prediction == 'Overweight_Level_I':
        st.error('You are in Overweight Level I category :thumbsdown:')
    if prediction == 'Overweight_Level_II':
        st.error('You are in Overweight Level II category :thumbsdown:')
    if prediction == 'Obesity_Type_I':
        st.warning('You are in Obesity Type I category :thumbsdown:')
    if prediction == 'Obesity_Type_II':
        st.warning('You are in Obesity Type II category :thumbsdown:')
    if prediction == 'Obesity_Type_III':
        st.warning('You are in Overweight Type III category :thumbsdown:')
    if prediction == 'Normal_Weight':
        st.success('You are in Normal Weight :thumbsup:')
    if prediction == 'Insufficient_Weight':
        st.success('Insufficient Weight :thumbsdown:')

