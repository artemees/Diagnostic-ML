# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:40:23 2022

@author: user
"""

import pickle
from streamlit_option_menu import option_menu 

brain_stroke = pickle.load(open("C:/Users/user/Downloads/collective diagnostic predictive system/saved models/brain_stroke.sav,"))

cardiovascular = pickle.load(open("C:/Users/user/Downloads/collective diagnostic gitpredictive system/saved models/cardiovascular.sav"))

diabetes = pickle.load(open("C:/Users/user/Downloads/collective diagnostic predictive system/saved models/diabetes_model.sav"))


# Navigation Sidebar

with st.sidebar:
    selected = option_menu('Diagnostic Predictive System',
                           ['Stroke',
                            'Cardiovascular Disorder',
                            'Diabetes Mellitus'],
                           default_index = 0)
    
# Stroke Page
if (selected =='Stroke' ):
    # page title
    st.title('Stroke Machine Learning prediction')
    
if (selected =='Cardiovascular Disorder' ):
    # page title
    st.title('Cardiovascular Disorder Machine Learning prediction')
    
if (selected =='Diabetes Mellitus' ):
    # page title
    st.title('Diabetes Machine Learning prediction')