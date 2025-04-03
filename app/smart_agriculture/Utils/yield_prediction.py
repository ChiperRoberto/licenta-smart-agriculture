import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Incarcam modelele tunate salvate anterior
model_ct_path = 'models/best_model_Yield_of_CT_Random_Forest.pkl'
model_nt_path = 'models/best_model_Yield_of_NT_Random_Forest.pkl'

model_ct = joblib.load(model_ct_path)
model_nt = joblib.load(model_nt_path)

st.title("Predicție randament cultură")
st.write("Introdu datele pentru a prezice randamentul estimat pentru sistemele CT și NT.")

# Încărcăm structura fișierului cu coloane pentru referință
reference_data = pd.read_csv("datasets/Processed_Database.csv")
feature_columns = reference_data.drop(columns=["Yield of CT", "Yield of NT"]).columns

# Colectăm inputuri de la utilizator pentru fiecare caracteristică
user_input = {}
st.sidebar.header("Datele parcelei")

for feature in feature_columns:
    if reference_data[feature].dtype in [np.float64, np.int64]:
        value = st.sidebar.number_input(f"{feature}", value=float(reference_data[feature].mean()))
    else:
        value = st.sidebar.selectbox(f"{feature}", options=sorted(reference_data[feature].unique()))
    user_input[feature] = value

# Convertim într-un DataFrame pentru predicție
input_df = pd.DataFrame([user_input])

# Predicție
if st.button("Prezice Randamentul"):
    try:
        yield_ct = model_ct.predict(input_df)[0]
        yield_nt = model_nt.predict(input_df)[0]

        st.success("Predicție completă!")
        st.metric(label="Randament CT (kg/ha)", value=f"{yield_ct:,.2f}")
        st.metric(label="Randament NT (kg/ha)", value=f"{yield_nt:,.2f}")

        better = "CT" if yield_ct > yield_nt else "NT"
        st.info(f"**Sistemul recomandat** pentru această parcelă este: `{better}`")
    except Exception as e:
        st.error(f"A apărut o eroare la predicție: {e}")
