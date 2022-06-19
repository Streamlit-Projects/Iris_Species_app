import streamlit as st
import streamlit.components.v1 as stc
from eda_app import run_eda_app
from ml_app import run_ml_app

st.title("Classifying Iris Species")

def main():
    #stc.html(html_temp)
    menu = ["About","EDA","ML"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "About":
        st.subheader("About")
        st.write("""
        Iris dataset is perhaps the best known database to be found in the pattern recognition literature.
        The data set contains 3 classes of 50 instances each, where each class refers to a type of iris
        plant.
        
        Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class: Iris Setosa, Iris Versicolour, Iris Virginica
        ##### Datasource
        The dataset used can be found on  UC Irvine Machine Learning Repository website:
        - http://archive.ics.uci.edu/ml/datasets/Iris
        ##### App Content
        - EDA Section: Exploratory Data Analysis of Data
        - ML Section: ML Predictor App (using k-NN)
        """)
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()

if __name__ == '__main__':
    main()
