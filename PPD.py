from copyreg import pickle
from doctest import OutputChecker
from importlib.abc import PathEntryFinder
from unittest import result
from PIL import Image
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import scipy
import pandas as pd
import math
import time
import plotly.express as px
import seaborn as sns
import shap
import xgboost as xgb  ###xgboost
from xgboost.sklearn import XGBClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability prediction of postpartum depression risk", layout="wide")

plt.style.use('default')

df=pd.read_csv('traindata1.csv',encoding='utf8')
trainy=df.PPD
trainx=df.drop('PPD',axis=1)
xgb = XGBClassifier(colsample_bytree=0.5,gamma=0.1,learning_rate=0.01,max_depth=4,iteration_range=200,
                    n_estimators =200,min_child_weight=4,subsample=1,verbosity = 0,verbose=3,
                    objective= 'binary:logistic',random_state = 1,class_weight = {0:0.82,1:0.18})
xgb.fit(trainx,trainy)

###side-bar
def user_input_features():
    st.title("Probability prediction of postpartum depression risk")
    st.sidebar.header('User input parameters below')
    a1=st.sidebar.slider("Prenatal score of EPDS",0,30)
    a2=st.sidebar.selectbox('Mother-in-law problems',('No','Yes'))
    a3=st.sidebar.selectbox('Domestic violence',('No','Yes'))
    a4=st.sidebar.selectbox('Mood',('Good','Moderate','Poor'))
    a5=st.sidebar.selectbox('Pressure',('Mild','Moderate','Severe'))
    a6=st.sidebar.selectbox('Prenatal self-harm tendency',('No','Yes'))
    result=""
    if a2=="Yes":
        a2=1
    else: 
        a2=0 
    if a3=="Yes":
        a3=1
    else: 
        a3=0 
    if a6=="Yes":
        a6=1
    else: 
        a6=0 
    if a4=="Good":
        a4=0
    elif a4== 'Moderate':
        a4=1
    else: 
        a4=2
    if a5=='Mild':
        a5=0
    elif a5== 'Moderate':
        a5=1
    else: 
        a5=2
    output=[a1,a2,a3,a4,a6,a5]
    int_features=[int(x) for x in output]
    final_features=np.array(int_features)
    patient1=pd.DataFrame(output)
    patient=pd.DataFrame(patient1.values.T,columns=trainx.columns)
    prediction=xgb.predict_proba(patient)
    prediction=float(prediction[:, 1])

    def predict_PPD():
        prediction=round(user_input_features[:, 1],3)
        return prediction

    result=""
    if st.button("Predict"):
        st.success('The probability of PPD for the mother: {:.1f}%'.format(prediction*100))
        if prediction>0.215:
            b="High risk"
        else:
            b="Low risk"
        st.success('The risk group:'+ b)
        explainer_xgb = shap.TreeExplainer(xgb)
        shap_values= explainer_xgb(patient)
        shap.plots.waterfall(shap_values[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Waterfall plot analysis of PPD for the mother:")
        st.pyplot(bbox_inches='tight')
        st.write("Abbreviations: EPDS, edinburgh postnatal depression scale; PPD, postpartum depression")
    if st.button("Reset"):
        st.write("")
    st.markdown("*Statement: this website will not record or store any information inputed.")
    st.write("2022 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
    st.write("✉ Contact Us: zoujianjun100@126.com")

if __name__ == '__main__':
    user_input_features()
