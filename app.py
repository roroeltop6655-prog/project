import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_assets():
    with open('student_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    return model, le_dict

model, le_dict = load_assets()

st.title("🎓 Student Success Prediction System")


st.sidebar.header("📝 Enter Student Data")

def user_input_features():
    age = st.sidebar.slider("Age", 15, 22, 18)
    studytime = st.sidebar.slider("Weekly Study Time (Hours)", 1, 4, 2, help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
    failures = st.sidebar.selectbox("Number of past class failures", [0, 1, 2, 3, 4])
    absences = st.sidebar.number_input("Number of school absences", 0, 100, 5)
    health = st.sidebar.slider("Current health status (1-5)", 1, 5, 3)
    freetime = st.sidebar.slider("Free time after school (1-5)", 1, 5, 3)
    goout = st.sidebar.slider("Going out with friends (1-5)", 1, 5, 3)
    traveltime = st.sidebar.slider("Home to school travel time (1-4)", 1, 4, 1)
    Medu = st.sidebar.selectbox("Mother's education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary", "Lower Secondary", "Higher Secondary", "Higher Education"][x])
    Fedu = st.sidebar.selectbox("Father's education", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Primary", "Lower Secondary", "Higher Secondary", "Higher Education"][x])

    sex = st.sidebar.radio("Gender", ["F", "M"], format_func=lambda x: "Female" if x=="F" else "Male")
    address = st.sidebar.radio("Address Type", ["U", "R"], format_func=lambda x: "Urban" if x=="U" else "Rural")
    famsize = st.sidebar.radio("Family Size", ["GT3", "LE3"], format_func=lambda x: "Greater than 3" if x=="GT3" else "3 or less")
    Pstatus = st.sidebar.radio("Parent's Cohabitation Status", ["T", "A"], format_func=lambda x: "Living Together" if x=="T" else "Apart")
    schoolsup = st.sidebar.checkbox("Extra educational support")
    famsup = st.sidebar.checkbox("Family educational support")
    paid = st.sidebar.checkbox("Extra paid classes")
    activities = st.sidebar.checkbox("Extra-curricular activities")
    higher = st.sidebar.checkbox("Wants to take higher education", value=True)
    internet = st.sidebar.checkbox("Internet access at home", value=True)
    romantic = st.sidebar.checkbox("In a romantic relationship")

    data = {
        'age': age, 'Medu': Medu, 'Fedu': Fedu, 'traveltime': traveltime,
        'studytime': studytime, 'failures': failures, 'absences': absences,
        'health': health, 'freetime': freetime, 'goout': goout,
        'sex': sex, 'address': address, 'famsize': famsize, 'Pstatus': Pstatus,
        'schoolsup': 'yes' if schoolsup else 'no',
        'famsup': 'yes' if famsup else 'no',
        'paid': 'yes' if paid else 'no',
        'activities': 'yes' if activities else 'no',
        'higher': 'yes' if higher else 'no',
        'internet': 'yes' if internet else 'no',
        'romantic': 'yes' if romantic else 'no'
    }
    return pd.DataFrame([data])

input_df = user_input_features()

processed_df = input_df.copy()
for col, le in le_dict.items():
    processed_df[col] = le.transform(processed_df[col])

st.subheader("📊 Prediction Result")
if st.button("Predict Result"):
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.success("🎉 The student is predicted to Pass!")
        else:
            st.error("⚠️ The student is predicted to Fail or needs extra support.")
            
    with col2:
        st.metric("Success Probability", f"{prediction_proba[0][1]*100:.1f}%")
        st.metric("Failure Probability", f"{prediction_proba[0][0]*100:.1f}%")
    st.write("---")
    st.subheader("📊 Probability Analysis Chart")
    
    chart_data = pd.DataFrame({
        'Status': ['Success', 'Failure'],
        'Probability (%)': [prediction_proba[0][1] * 100, prediction_proba[0][0] * 100]
    })
    
    st.bar_chart(data=chart_data, x='Status', y='Probability (%)', color=['#2ecc71'])
    # Display Input Data
    st.write("---")
    st.subheader("📋 Input Summary")
    st.write(input_df)
else:
    st.info("Click the 'Predict Result' button to get the analysis.")

st.markdown("---")
st.markdown("The Project Team:  \n"
"Hafsa Ahmed, "
"Arwa Ahmed, "
"Ayisha Ahmed, "
"Menna Moustafa, "
"Simon Rafaat, "
"Esraa Kamal, "
"Amira Waled")
