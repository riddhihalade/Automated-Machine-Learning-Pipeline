import streamlit as st
import pandas as pd
from pycaret.regression import setup, compare_models, save_model, pull
import os
import sweetviz as sv


if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv')
else:
    df = None

# Sidebar navigation and homepage content
with st.sidebar: 
    st.image("ml.png")
    st.title("Automated ML")
    st.info("This application helps you explore your data and train it for desired results with automated ML pipeline.")

# Main content based on user choice
choice = st.sidebar.radio("Navigation", ["Home","Upload", "Profiling", "Modelling", "Download"])
# Homepage content
if choice == "Home":
    st.title("ModelForge: Automated ML Hub")
    st.markdown("""
    Welcome to the ModelForge an Automated ML application!  
    This application helps you explore your data and train it for desired results with automated ML pipeline.
    It finds you the best model suited for your dataset.

    Choose an option from the navigation to get started:
    - **Upload**: Upload your dataset for analysis.
    - **Profiling**: Perform Exploratory Data Analysis (EDA) to understand your data.
    - **Modelling**: Train machine learning models and compare their performance.
    - **Download**: Download the best performing model for deployment.
    """)



if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=['csv'])
    if file: 
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

if choice == "Profiling" and df is not None: 
    st.title("Exploratory Data Analysis")
    report = sv.analyze(df)
    report.show_html(filepath='report.html', open_browser=False)
    with open('report.html', 'r') as f:
        html = f.read()
    st.components.v1.html(html, height=1700, width=1000)
elif choice == "Profiling" and df is None:
    st.warning("Please upload a dataset first.")
    
elif choice == "Profiling" and df is None:
    st.warning("Please upload a dataset first.")

if choice == "Modelling" and df is not None: 
    st.title("Get the Best Suited Model")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(data=df, target=chosen_target, verbose=False)
        setup_df = pull()
        st.dataframe(setup_df)
        
        best_model = compare_models()
        st.write("Best Model:", best_model)
        
        st.write("Check out the Model's Comparative Analysis")
        compare_df = pull()
        st.dataframe(compare_df)

        save_model(best_model, 'best_model')
        st.write("Best Model saved!")
elif choice == "Modelling" and df is None:
    st.warning("Please upload a dataset first.")
    
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

