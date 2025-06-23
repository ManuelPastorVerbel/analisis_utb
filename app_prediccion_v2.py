
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Dashboard Emprendimiento", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("modelo_emprendimiento_resultados.csv")
    return df

df = cargar_datos()

# Título
st.title("📊 Dashboard: Emprendimiento y Predicción de Formalización")
st.markdown("Explora los perfiles socioeconómicos y predice la probabilidad de inscripción en Cámara de Comercio.")

# Filtros
st.sidebar.header("🎛️ Filtros")
sexo = st.sidebar.multiselect("Sexo", df["Sexo"].unique(), default=df["Sexo"].unique())
nivel = st.sidebar.multiselect("Nivel Educativo", df["Nivel_Educativo"].unique(), default=df["Nivel_Educativo"].unique())
sector = st.sidebar.multiselect("Sector Económico", df["Sector_Economico"].unique(), default=df["Sector_Economico"].unique())

df_filtrado = df[
    (df["Sexo"].isin(sexo)) &
    (df["Nivel_Educativo"].isin(nivel)) &
    (df["Sector_Economico"].isin(sector))
]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total registros", len(df_filtrado))
col2.metric("✅ % Inscritos", f"{df_filtrado['Inscrito_Camara_Comercio'].mean()*100:.1f}%")
col3.metric("📈 Promedio probabilidad", f"{df_filtrado['Probabilidad_Inscripcion'].mean():.2f}")
col4.metric("⚠️ Riesgo ALTO", f"{(df_filtrado['Riesgo_Desercion'] == 'ALTO').mean()*100:.1f}%")

# Gráficos principales
st.subheader("📌 Distribuciones clave")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(data=df_filtrado, x="Edad", hue="Sexo", multiple="stack", ax=ax, bins=20)
    ax.set_title("Distribución de Edad por Sexo")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(data=df_filtrado, x="Nivel_Educativo", hue="Sexo", ax=ax)
    ax.set_title("Nivel Educativo por Sexo")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("🎯 Riesgo de Deserción")
col1, col2 = st.columns([1, 2])
with col1:
    riesgo_counts = df_filtrado["Riesgo_Desercion"].value_counts().reindex(["ALTO", "MEDIO", "BAJO"])
    fig, ax = plt.subplots()
    riesgo_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=["red", "orange", "green"])
    ax.set_ylabel("")
    ax.set_title("Distribución de Riesgo")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(data=df_filtrado, x="Sector_Economico", y="Probabilidad_Inscripcion", hue="Riesgo_Desercion", ax=ax)
    ax.set_title("Probabilidad promedio por sector y riesgo")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Predicción individual
st.subheader("🤖 Predicción personalizada")

with st.form("form_prediccion"):
    edad = st.number_input("Edad", min_value=15, max_value=90, value=30)
    sexo_i = st.selectbox("Sexo", df["Sexo"].unique())
    nivel_i = st.selectbox("Nivel Educativo", df["Nivel_Educativo"].unique())
    estado_i = st.selectbox("Estado Civil", df["Estado_Civil"].unique())
    sector_i = st.selectbox("Sector Económico", df["Sector_Economico"].unique())
    submitted = st.form_submit_button("Predecir")

    if submitted:
        input_dict = {
            "Edad": edad,
            "Sexo": sexo_i,
            "Nivel_Educativo": nivel_i,
            "Estado_Civil": estado_i,
            "Sector_Economico": sector_i
        }
        input_df = pd.DataFrame([input_dict])
        df_dummies = pd.get_dummies(df.drop(columns=["Inscrito_Camara_Comercio", "Probabilidad_Inscripcion", "Prediccion_Modelo", "Riesgo_Desercion"]), drop_first=True)
        X = df_dummies
        y = df["Inscrito_Camara_Comercio"]
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        input_dummies = pd.get_dummies(input_df, drop_first=True)
        for col in X.columns:
            if col not in input_dummies.columns:
                input_dummies[col] = 0
        input_dummies = input_dummies[X.columns]

        prob = modelo.predict_proba(input_dummies)[0][1]
        riesgo = "ALTO ❌" if prob < 0.3 else "MEDIO ⚠️" if prob <= 0.6 else "BAJO ✅"

        st.success(f"📈 Probabilidad de inscripción: {prob:.2%}")
        st.info(f"🛑 Riesgo de deserción: {riesgo}")
