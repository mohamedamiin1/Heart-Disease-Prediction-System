# -*- coding: utf-8 -*-

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# loading the saved models
heart_disease_model = pickle.load(open('C:/Users/Administrator/Desktop/Heart Disease Prediction System/saved_models/heart_disease_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Heart Disease Prediction System',

                           [
                            'Home',
                            'Prediction',
                            'Dataset',
                            'About Us'
                            ],
                           menu_icon='heart-fill',
                           icons=['house', 'book', 'envelope', 'list'],
                           default_index=0)
# Home Page
if selected == 'Home':

    st.markdown("""
    <style>
    .big-font1 {
        font-size: 50px !important;
        color: red;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<p class='big-font1'>Heart Disease</p>", unsafe_allow_html=True)

    # Open the GIF image file and read its contents
    with open("C:\\Users\\Administrator\\Desktop\\Heart Disease Prediction System\\Images\\Heart\\heartbeating1.gif", "rb") as file:
        contents = file.read()

    # Encode the image file content to base64
    data_url = base64.b64encode(contents).decode("utf-8")

    # Display the image using HTML img tag with the encoded data URL
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" width="700" alt="heart gif">',
        unsafe_allow_html=True
    )

    st.write("Heart Disease is one of the leading causes of death in the world, in fact in the United States this disease is there leading cause of death their heart failure is commonly known as (CAD) or coronary artery disease. The term 'heart disease' refers to several type of heart condition, a type of disease that affects the heart or blood vessels. So our Heart Disease Prediction System, is a powerful tool designed to assess the risk of heart disease using advanced machine learning techniques. By leveraging Logistic Regression, our system analyzes critical medical parameters such as age, gender, blood pressure, cholesterol levels, and more to provide an accurate prediction of heart disease risk. This system aims to support healthcare professionals and individuals in early detection and proactive management of heart health, ultimately contributing to better health outcomes.")

   
   










# Heart Disease Prediction Page
if selected == 'Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)






 
        
 
# Dataset page

df = pd.read_csv('C:/Users/Administrator/Desktop/Heart Disease Prediction System/dataset/heart.csv')
df = df.dropna()


if selected == 'Dataset':
    st.markdown("""
    <style>            
        .big-font1 {
            font-size: 50px !important;
            color: red;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<p class='big-font1'>Data Analysis</p>", unsafe_allow_html=True)
    
    if st.checkbox("Display Dataset"): 
        st.dataframe(df.head(20))

    if st.checkbox("Display Rows and Column Shape"): 
        st.write(df.shape)

    if st.checkbox("Display columns"): 
        st.write(df.columns) 
    
    if st.checkbox("Select multiple columns"): 
        selected_columns = st.multiselect('Select preferred columns:', df.columns) 
        df1 = df[selected_columns] 
        st.dataframe(df1) 
    
    if st.checkbox("Display summary"): 
        st.write(df1.describe().T) 
 # Ensure your DataFrame df1 is defined earlier in the code       
    if st.checkbox("Display Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df1.corr(), linewidths=1.0, ax=ax)
        st.pyplot(fig)

 



# About Us page
if selected == 'About Us':
    st.title("About Us")
    st.write("One of our goals why we create this heart disease prediction is to determine your rate of getting this disease by some measurements that the predictions required to you to fill up. Also, minimize the rate of having this deadly disease.")

    st.subheader("Members")
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open ('C:/Users/Administrator/Desktop/Heart Disease Prediction System/Images/me.png')
        st.image(image, caption="Mohamed Abdiwahab Mohamed.")
    with col2:
        image = Image.open ('C:/Users/Administrator/Desktop/Heart Disease Prediction System/Images/osman.png')
        st.image(image, caption="Osman Hassan Mohamud.")
    st.write("---")
    st.subheader("Supervisor")
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open ('C:/Users/Administrator/Desktop/Heart Disease Prediction System/Images/ustad.png')
        st.image(image, caption="Eng. Abdifatah Farah Ali.")
    with col2:
        st.write("We are immensely grateful to our Supervisor Eng. Abdifitah Farah Ali, ICT director at SIMAD University. for his exceptional guidance, technical expertise, and unwavering commitment throughout the our project. His insightful feedback, mentorship, and ability to foster a collaborative environment were instrumental to the success of this endeavor. We are deeply honored to have had the opportunity to work under your supervision and look forward to future collaborations that will undoubtedly benefit from your invaluable expertise.") 
        
        

