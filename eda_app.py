import streamlit as st
import numpy as np
import pandas as pd
import os.path

# Imports for plotly:
import plotly.express as px


@st.cache(allow_output_mutation=True)
def load_data(data_path):
    dir_name = os.path.abspath(os.path.dirname(__file__))
    location = os.path.join(data_path, 'Iris.csv')
    df = pd.read_csv(location)
    return df

def spc_id(i):
    if i == 'Iris-setosa':
        return 1
    elif i == 'Iris-versicolor':
        return 2
    else:
        return 3

DATA_PATH = ("data")

def run_eda_app():
    st.subheader("Exploratory Data Analysis Section")
    df = load_data(DATA_PATH)
    df['Species ID'] = df['Species'].apply(spc_id)

    bar_df = pd.DataFrame(df.groupby(['Species'])['Species'].count())
    bar_df = bar_df.rename(columns = {"Species":"Volume"}).reset_index()
    
    def desc(df):
        d = pd.DataFrame(df.dtypes,columns=['Data Types'])
        d = d.reset_index()
        d['Columns'] = d['index']
        d = d[['Columns','Data Types']]
        d['Missing'] = df.isnull().sum().values
        d['Uniques'] = df.nunique().values
        return d

    submenu = st.sidebar.selectbox("SubMenu",["Plots", "Descriptive"])
    if submenu == "Plots":
        st.subheader("Plots")
        st.markdown("###### In this section we will use Exploratory Data Analysis (EDA) to find out more about iris dataset via data visualisations.")
        # Species distribution visuals (bar & donut)
        st.sidebar.markdown("#### Data Distribution of Iris Species")
        viz_type = st.sidebar.selectbox('Visualization type', ['Donut chart', 'Bar plot'], key='1')
        if not st.sidebar.checkbox("Close", True, key='1'):
            st.markdown("##### **Data Distribution of Iris Species**")
            if viz_type == 'Bar plot':
                st.markdown("###### **Bar plot**")
                st.markdown("###### The iris dataset contains 50 samples (rows) for each of the species. Dataset is perfectly balanced.")
                fig_bar = px.bar(bar_df, x='Species', y='Volume', color='Species', color_discrete_sequence=['#0e9aa7', '#f6cd61', '#fe8a71'])
                st.plotly_chart(fig_bar)
            if viz_type == 'Donut chart':
                st.markdown("###### **Donut chart**")
                st.markdown("###### The iris dataset contains 50 samples (rows) for each of the species. Dataset is perfectly balanced.")
                fig_don = px.pie(bar_df, values='Volume', names='Species', hole=0.45, color_discrete_sequence=['#0e9aa7', '#f6cd61', '#fe8a71'])
                st.plotly_chart(fig_don)
                
        # Scatter plots (petals & sepals)
        st.sidebar.markdown("#### Relationships between Features")
        if not st.sidebar.checkbox("Close", True):
            x_value = st.sidebar.selectbox('x axis:', ['Petal length', 'Sepal length'], key='2')
            y_value = st.sidebar.selectbox('y axis:', ['Petal width', 'Sepal width'], key='3')
            
            st.markdown("##### **Relationships between lenght & width**")
            st.markdown("###### **Scatter plot**")
            st.markdown("###### In this section we will plot data specifically for Sepal/Petal lengt and Sepal/Petal width. We can find all scatter plot combinations by selecting desired categories for x and y axis on a sidebar.")
            fig_scatter = px.scatter(df, x=x_value, y=y_value, color="Species", color_discrete_sequence=['#0e9aa7', '#f6cd61', '#fe8a71'])
            st.plotly_chart(fig_scatter)
        
        # Distribution of Lenght & Width (Histograms and Box Plots)
        st.sidebar.markdown("#### Distribution of Features")
        dist_type = st.sidebar.selectbox('Visualization type', ['Histogram', 'Box Plot'], key='4')
        dist_select = st.sidebar.selectbox('Select category', ['Petal length', 'Sepal length', 'Petal width', 'Sepal width'], key='5')
        if not st.sidebar.checkbox("Close", True, key='4'):
            st.markdown("##### **Distribution of Lenght and Width**")
            if dist_type == 'Histogram':
                st.markdown("###### **Histogram**")
                st.markdown("###### Please find below histogram for Sepal and Petal dimensions:")
                fig_hist = px.histogram(df, x=dist_select, color="Species", color_discrete_sequence=['#0e9aa7', '#f6cd61', '#fe8a71'])
                st.plotly_chart(fig_hist)
            if dist_type == 'Box Plot':
                st.markdown("###### **Box Plot**")
                st.markdown("###### Please find below box plots for Sepal and Petal dimensions:")
                fig_box = px.box(df, x="Species", y=dist_select, color="Species", color_discrete_sequence=['#0e9aa7', '#f6cd61', '#fe8a71'])
                st.plotly_chart(fig_box)
                
        # Iris Spieces comparison (Parallel Coordinates)
        st.sidebar.markdown("#### Iris Spiecies Comparison")
        comp_select = st.sidebar.selectbox('Visualization type', ['Parallel Coordinates', 'Facet Plot'], key='6')
        if not st.sidebar.checkbox("Close", True, key='6'):
            st.markdown("##### **Iris Spiecies Comparison**")
            if comp_select == 'Parallel Coordinates':
                st.markdown("###### **Parallel Coordinates**")
                st.markdown("###### To make it easy to read the species key is as follows: 1 setosa, 2 versicolor and 3 virginica")
                fig = px.parallel_coordinates(df
                                              , color='Species ID'
                                              , labels={'Spiecies ID':'Species'
                                                        ,'Sepal width':'Sepal width'
                                                        ,'Sepal length':'Sepal length'
                                                        ,'Petal width':'Petal width'
                                                        ,'Petal length':'Petal length'}
                                              , color_continuous_scale = ['#0e9aa7', '#f6cd61', '#fe8a71']
                                              , color_continuous_midpoint=2
                                             )
                st.plotly_chart(fig)
            if comp_select == 'Facet Plot':
                st.markdown("###### **Facet Plot**")
                st.markdown("###### We are comparing Sepal lenght and Petal width/lenght for each of the species.")
                fig1 = px.bar(df, x="Sepal length", y="Petal width", color = 'Petal length', facet_row="Species", height=700)
                st.plotly_chart(fig1)
                
    else:
        st.dataframe(df)
        
        with st.expander("Data Description"):
            st.dataframe(df.describe().transpose())
            
        with st.expander("Data Types Overview"):
            st.dataframe(desc(df).astype(str))
            
        with st.expander("Class Distribution"):
            st.dataframe(df['Species'].value_counts())         		
