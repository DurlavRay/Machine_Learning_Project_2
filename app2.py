import streamlit as st
import pickle
st.title('TAXIS ML Project')
distance = st.number_input('Distance', value=20, placeholder='enter a value for distance')
fare = st.number_input('Fare', value=10, placeholder='enter a value for fare')
tip = st.number_input('Tip', value=5, placeholder='enter a value for tip')
loaded_model = pickle.load(open('taxis_regression.sav', 'rb'))
prediction = loaded_model.predict([[distance,fare,tip]])
st.subheader(f'predicted total value for above parameter is {prediction[0]}')
st.write(prediction)