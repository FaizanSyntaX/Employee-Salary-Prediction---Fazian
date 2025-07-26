import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="💼 Salary Prediction App", page_icon="💸", layout="centered")

# 💼 Title and Description
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>💼 Employee Salary Prediction</h1>
    <p style='text-align: center;'>Predict if an employee earns more than 50K using demographic and job-related details.</p>
    <hr style="border:1px solid #ccc">
""", unsafe_allow_html=True)

# 📊 Sidebar - Input Fields
st.sidebar.header("📥 Enter Employee Details")

# Input fields
age = st.sidebar.slider("🧓 Age", 18, 70, 30)
workclass = st.sidebar.selectbox("🏢 Workclass", encoders['workclass'].classes_)
fnlwgt = st.sidebar.number_input("⚖️ Final Weight (fnlwgt)", value=100000)
education_num = st.sidebar.slider("🎓 Education Level (numeric)", 1, 16, 10)
marital_status = st.sidebar.selectbox("💍 Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("💼 Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("👥 Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("🌎 Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("⚧️ Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("📈 Capital Gain", value=0)
capital_loss = st.sidebar.number_input("📉 Capital Loss", value=0)
hours_per_week = st.sidebar.slider("⏱️ Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("🗺️ Native Country", encoders['native-country'].classes_)

# Encode categorical inputs
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [encoders['workclass'].transform([workclass])[0]],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [encoders['marital-status'].transform([marital_status])[0]],
    'occupation': [encoders['occupation'].transform([occupation])[0]],
    'relationship': [encoders['relationship'].transform([relationship])[0]],
    'race': [encoders['race'].transform([race])[0]],
    'gender': [encoders['gender'].transform([gender])[0]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [encoders['native-country'].transform([native_country])[0]],
})

st.markdown("### 🔍 Preview of Input Data")
st.dataframe(input_data)

# 🔮 Prediction Button
if st.button("🔎 Predict Salary Class"):
    pred = model.predict(input_data)[0]
    result = "💰 >50K (High Income)" if pred == ">50K" else "🧾 <=50K (Low Income)"
    color = "#2ECC71" if pred == ">50K" else "#E74C3C"
    st.markdown(f"""
        <div style='text-align: center; background-color: {color}; padding: 1rem; border-radius: 8px; color: white; font-size: 1.5rem;'>
            ✅ Prediction: {result}
        </div>
    """, unsafe_allow_html=True)

# 📂 Batch Prediction Section
st.markdown("---")
st.markdown("### 📁 Batch CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file)  # Unencoded copy for display
    df = df_original.copy()                   # This one will be encoded

    df = df.dropna()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    try:
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
    except ValueError as e:
        st.error(f"❌ Encoding failed: {e}")
        st.stop()

    # Predict using encoded data
    batch_preds = model.predict(df)

    # Map predictions back to label
    label_map = {0: "<=50K", 1: ">50K"}
    df_original['PredictedClass'] = [label_map[p] for p in batch_preds]

    st.success("✅ Batch Prediction Completed")
    st.write(df_original)

    csv = df_original.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
