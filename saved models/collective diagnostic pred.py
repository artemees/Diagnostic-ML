# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:40:23 2022

@author: user
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu 

path = 'C:/Users/user/Downloads/collective diagnostic predictive system'

brain_stroke = pickle.load(open("C:/Users/user/Downloads/collective diagnostic predictive system/saved models/brain_stroke.sav","rb"))


cardiovascular = pickle.load(open("C:/Users/user/Downloads/collective diagnostic predictive system/saved models/cardiovascular.sav","rb"))

diabetes = pickle.load(open("C:/Users/user/Downloads/collective diagnostic predictive system/saved models/diabetes_model.sav","rb"))


# Navigation Sidebar

with st.sidebar:
    selected = option_menu('Diagnostic Predictive System Using Machine Learning.',
                           
                           ['Stroke',
                            'Cardiovascular Disorder',
                            'Diabetes Mellitus'],
                           
                           icons =['activity','heart','person-rolodex'],
                           
                           default_index = 0)
    
# Stroke Page
if (selected =='Stroke' ):
    # page title
    st.title('Diagnostic Prediction: Stroke')
    
    gender = st.text_input('Gender: Male[1] - Female[0]')
    age = st.text_input('Age')
    hypertension = st.text_input('Hypertension: Yes[1]  - No[0]')  
    heart_disease = st.text_input('Heart Disease: Yes[1]-  No[0]')
    ever_married = st.text_input('Ever Married: Yes[1] - No[0]')
    work_type = st.text_input('Work Type: Private[1] |  Self-employed[2] |  Govt_job[3] |  Child[4]')
    Residence_type = st.text_input('Residential Location: Urban[1] - Rural[0]')
    avg_glucose_level = st.text_input('Glucose Level: Input Value')
    bmi = st.text_input('Body Mass Index(BMI): Input Value')
    smoking_status = st.text_input('Smoking Status: Formerly[1] |  Never[2] | Smokes[3] | Uncertain[4]')
    
    # Prediction
    stroke_diagnosis = ''
    
    # Prediction Button
    
    if st.button("Diagnose Me!"):
        stroke_prediction = brain_stroke.predict([[gender,age,hypertension,heart_disease,ever_married,work_type, Residence_type,avg_glucose_level,bmi,smoking_status]])
        
        if(stroke_prediction[0]==0):
            stroke_diagnosis = 'Not at Risk'
            
        else:
            stroke_diagnosis = 'Person has had / is at risk of having Stroke.'
            
    
    st.success(stroke_diagnosis)
    
    
    #user Inputed data
    
    #col1. col2, col3 = st.columns(3)
    
    #with col1:
        #gender = st.text_input('Gender')
        
    #with col2:
       # age = st.text_input('Age')
        
#        hypertension = st.text_input('Hypertension Present(1)/Absent(0)')    
        
    #with col1:
       # heart_disease = st.text_input('Heart Disease Present(1)/Absent(0)')
        
    #with col2:
       # ever_married = st.text_input('Married Yes(1)/No(0)')
        
    #with col3:
        #work_type = st.text_input('Private(1)/self-employed(2)/Govt_job(3),Child(4)')
             
    #with col1:
        #Residence_type = st.text_input('Residence_type Urban(1)/Rural(0)')
                
    #with col2:
        # avg_glucose_level = st.text_input('Glucose Level')
        
    #with col3:
        # bmi = st.text_input('Body Mass Index(BMI)')    
         
    #with col1:
        #smoking_status = st.text_input('Smoking Status Formerly(1)/Never(2)/Smokes(3)/Uncertain(4)')    
        
        
        
    
if (selected =='Cardiovascular Disorder' ):
    # page title
    st.title('Diagnostic Prediction: Cardiovascular Disorder')
    
if (selected =='Diabetes Mellitus' ):
    # page title
    st.title(' Diagnostic Prediction: Diabetes Mellitus')