import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoded_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoded_gender = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

model = load_model('model.h5')

st.title('Customer Churn Prediction')

# User Input 
credit_score = st.number_input('CreditScore')
geography = st.selectbox('Geography',onehot_encoded_geo.categories_[0])
gender = st.selectbox('Gender',label_encoded_gender.classes_)
age = st.slider('Age',18,92)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
num_of_products = st.slider('NumOfProducts',1,4)
has_cr_card = st.selectbox('HasCrCard',[0,1])
is_active_member = st.selectbox('IsActiveMember',[0,1])
estimated_salary = st.number_input('EstimatedSalary')

# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoded_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoded_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoded_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaled = scalar.transform(input_df)

if st.button('Predict Churn'):
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    st.write(f"Churn Probability: {prediction_proba:.2f}")
    if prediction_proba > 0.5:
        st.write("The person is likely to churn")
    else:
        st.write("The person is not likely to churn")