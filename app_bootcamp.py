

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Proyecto Emprendimiento - Barranquilla", layout="wide", page_icon=" ")

st.title(" Proyecto de Emprendimiento - Barranquilla")
vista = st.sidebar.radio("Selecciona una vista:", ["📊 Análisis Exploratorio", "🤖 Predicción Individual"])

@st.cache_data
def cargar_datos():
    df = pd.read_csv("modelo_emprendimiento_resultados.csv")
    df["Sexo"] = df["Sexo"].astype(str).str.upper().str.strip()
    df = df[df["Sexo"].isin(["F", "M"])]
    df["Nivel_Educativo"] = df["Nivel_Educativo"].astype(str).str.upper().str.strip()
    return df

df = cargar_datos()

if vista == "📊 Análisis Exploratorio":
    st.header("📊 Análisis de Intención Emprendedora - Barranquilla 2025")
    st.markdown("Seleccione filtros para explorar los factores clave del emprendimiento.")

    sexo = st.sidebar.multiselect("Sexo", ["F", "M"], default=["F", "M"])
    sector = st.sidebar.multiselect("Sector Económico", df["Sector_Economico"].unique(), default=df["Sector_Economico"].unique())
    nivel = st.sidebar.multiselect("Nivel Educativo", df["Nivel_Educativo"].unique(), default=df["Nivel_Educativo"].unique())

    df_filtrado = df[
        (df["Sexo"].isin(sexo)) &
        (df["Sector_Economico"].isin(sector)) &
        (df["Nivel_Educativo"].isin(nivel))
    ]

    st.subheader("1️⃣ Perfil socioeconómico con intención emprendedora (Inscritos)")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtrado[df_filtrado["Inscrito_Camara_Comercio"] == 1], x="Nivel_Educativo", hue="Sexo", ax=ax)
        ax.set_title("Nivel educativo por sexo (Inscritos)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_filtrado[df_filtrado["Inscrito_Camara_Comercio"] == 1], x="Sexo", y="Edad", ax=ax)
        ax.set_title("Edad por sexo (Inscritos)")
        st.pyplot(fig)

    st.subheader("2️⃣ Diferencias entre mujeres y hombres con intención emprendedora")
    promedios = df_filtrado.groupby(["Sexo", "Nivel_Educativo"])["Probabilidad_Inscripcion"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=promedios, x="Sexo", y="Probabilidad_Inscripcion", hue="Nivel_Educativo", ax=ax)
    ax.set_title("Promedio de probabilidad por sexo y nivel educativo")
    ax.legend(title="Nivel Educativo", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("3️⃣ Relación entre sexo y sector económico")
    fig, ax = plt.subplots(figsize=(8,4))
    sexo_sector = pd.crosstab(df_filtrado["Sector_Economico"], df_filtrado["Sexo"], normalize="index") * 100
    sexo_sector.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("%")
    ax.set_title("Distribución de sexo por sector económico")
    st.pyplot(fig)

    st.subheader("4️⃣ Relación entre edad y sexo en emprendimiento")
    fig, ax = plt.subplots()
    sns.histplot(data=df_filtrado, x="Edad", hue="Sexo", multiple="stack", bins=20, ax=ax)
    ax.set_title("Distribución de edad por sexo")
    st.pyplot(fig)


else:
    st.header("🤖 Predicción de Inscripción en Cámara de Comercio")

    st.markdown("Ingresa los datos para predecir la probabilidad de inscripción y riesgo de deserción.")

    with st.form("form_prediccion"):
        edad = st.slider("Edad", 15, 90, 30)
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
            st.info(f"🛑 Nivel de riesgo: {riesgo}")
