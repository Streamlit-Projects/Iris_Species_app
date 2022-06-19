import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image


def run_ml_app():
    st.subheader("Machine Learning Section")
    
    # Loading our pretrained knn model:
    model= open("ml_models\knn_model.pkl", "rb")
    knn_model = joblib.load(model)
    
    # Load iris images:
    setosa = Image.open("iris_pictures\i_setosa.jpg")
    versicolor = Image.open("iris_pictures\i_versicolor.jpg")
    virginica = Image.open("iris_pictures\i_virginica.jpg")
    
    st.sidebar.markdown("#### Select Features:")
    parameter_list = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    parameter_input = []
    parameter_default = ['5.2', '3.2', '4.2', '1.2']

    for parameter, parameter_df in zip(parameter_list, parameter_default):
        values =st.sidebar.slider(label = parameter, key=parameter, value = float(parameter_df), min_value = 0.0, max_value = 8.0, step = 0.1)
        parameter_input.append(values)
        
    input_variables = pd.DataFrame([parameter_input], columns=parameter_list, dtype = float)
    st.write('\n\n')
    
    prediction = knn_model.predict(input_variables)
    prd = ' '.join(prediction)
    
    if st.button("Click Here to Classify"):
        st.write(prd)
        if prd == 'Iris-setosa':
            st.image(setosa)
        if prd == 'Iris-versicolor':
            st.image(versicolor)
        if prd == 'Iris-virginica':
            st.image(virginica)
