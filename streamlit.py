import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
import plotly .express as px

# st.title('Expresso churn')
# st.subheader('Built by Ifeyinwa')

Expresso = pd.read_csv('expresso_processed.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>CHURN PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by IFEYINWA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com.png')
st.divider()
st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("Customer churn is a critical challenge for businesses, impacting revenue and growth. The objective is to develop a predictive model that identifies customers at risk of leaving based on historical data, including demographics, engagement metrics, transaction history, and support interactions. By accurately predicting churn, businesses can take proactive measures, such as personalized offers or targeted engagement strategies, to improve retention and customer satisfaction. The goal is to minimize churn rate and maximize customer lifetime value.")
st.divider()

st.dataframe(Expresso,use_container_width= True)

st.sidebar.image('user icon churn.png',caption = "Welcome User")

datavolume = st.sidebar.number_input('Datavolume exp', min_value=-0.1, max_value=100000.0, value=Expresso.DATA_VOLUME.median())
onnet = st.sidebar.number_input('Onnet exp', min_value=-0.1, max_value=10000.0, value=Expresso.ON_NET.median())
regularity = st.sidebar.number_input('Regularity exp', min_value=-0.1, max_value=100.0, value=Expresso.REGULARITY.median())
revenue = st.sidebar.number_input('Revenue exp', min_value=0.0, max_value=40000.0, value=Expresso.REVENUE.median())
freq = st.sidebar.number_input('Freq exp', min_value= -0.1, max_value = 10.0, value=Expresso.FREQUENCE.median())
montant = st.sidebar.number_input('Montant exp', min_value=-0.1, max_value=100000.0, value=Expresso.MONTANT.median())
freqrech = st.sidebar.number_input('Frequecerech exp', min_value=-0.1, max_value=100000.0, value=Expresso.FREQUENCE_RECH.median())


# user input,we want to recognise the original name given in the dataset and link it

inputs = {
    'DATA_VOLUME' : [datavolume],
    'ON_NET' : [onnet],
    'REGULARITY' : [regularity],
    'REVENUE' : [revenue],
    'FREQUENCE' : [freq],
    'MONTANT' : [montant],    
   'FREQUENCE_RECH' : [freqrech],
}


# if we want the input  to appear under the  dataset

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

# transform the user inputs,import the transformers(scalers)

montant_scaler = joblib.load('MONTANT_scaler.pkl')
freq_scaler = joblib.load('FREQUENCE_scaler.pkl')
revenue_scaler = joblib.load('REVENUE_scaler.pkl')
datavolume_scaler = joblib.load('DATA_VOLUME_scaler.pkl')
regularity_scaler = joblib.load('REGULARITY_scaler.pkl')
onnet_scaler = joblib.load('ON_NET_scaler.pkl')
freqrech_scaler = joblib.load('FREQUENCE_RECH_scaler.pkl')


# link the scalers to the user inputs

inputVar['MONTANT'] = montant_scaler.transform(inputVar[['MONTANT']])
inputVar['FREQUENCE'] = freq_scaler.transform(inputVar[['FREQUENCE']])
inputVar['REVENUE'] = revenue_scaler.transform(inputVar[['REVENUE']])
inputVar['DATA_VOLUME'] = datavolume_scaler.transform(inputVar[['DATA_VOLUME']]) 
inputVar['REGULARITY'] = regularity_scaler.transform(inputVar[['REGULARITY']]) 
inputVar['ON_NET'] = onnet_scaler.transform(inputVar[['ON_NET']]) 
inputVar['FREQUENCE_RECH'] = freqrech_scaler.transform(inputVar[['FREQUENCE_RECH']]) 


#Bringing in the model
model = joblib.load('expresso_processed model.pkl')


# we create a button to use for the prediction

predictbutton = st.button('Push to Predict the CHURN')

if predictbutton: 
    predicted = model.predict(inputVar)
    st.success(f'the predicted Churn value is : {predicted}')
