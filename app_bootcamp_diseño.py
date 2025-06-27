
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



st.title("🚀 Proyecto de Emprendimiento - Barranquilla")

vista = st.sidebar.radio("Selecciona una vista:", ["📊 Análisis Exploratorio", "🤖 Predicción Individual"])

@st.cache_data
def cargar_datos():
    df = pd.read_csv("modelo_emprendimiento_resultados.csv")
    df["Sexo"] = df["Sexo"].str.upper().str.strip()
    df = df[df["Sexo"].isin(["F", "M"])]
    df["Nivel_Educativo"] = df["Nivel_Educativo"].str.upper().str.strip()
    return df

df = cargar_datos()

if vista == "📊 Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio del Emprendimiento")
    sexo = st.sidebar.multiselect("Sexo", ["F", "M"], default=["F", "M"])
    sector = st.sidebar.multiselect("Sector Económico", df["Sector_Economico"].unique(), default=df["Sector_Economico"].unique())
    nivel = st.sidebar.multiselect("Nivel Educativo", df["Nivel_Educativo"].unique(), default=df["Nivel_Educativo"].unique())

    df_filtrado = df[
        (df["Sexo"].isin(sexo)) &
        (df["Sector_Economico"].isin(sector)) &
        (df["Nivel_Educativo"].isin(nivel))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total registros", len(df_filtrado))
    col2.metric("% Inscritos", f"{df_filtrado['Inscrito_Camara_Comercio'].mean()*100:.1f}%")
    col3.metric("Prob. promedio", f"{df_filtrado['Probabilidad_Inscripcion'].mean():.2f}")

    st.subheader("1️⃣ Nivel educativo por sexo (inscritos)")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtrado[df_filtrado["Inscrito_Camara_Comercio"] == 1], x="Nivel_Educativo", hue="Sexo", ax=ax)
        ax.set_title("Nivel educativo por sexo")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_filtrado[df_filtrado["Inscrito_Camara_Comercio"] == 1], x="Sexo", y="Edad", ax=ax)
        ax.set_title("Edad por sexo")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("2️⃣ Diferencias por nivel educativo")
    # Normalizar niveles educativos
    df_filtrado["Nivel_Educativo"] = df_filtrado["Nivel_Educativo"].str.upper().str.strip()
    promedios = df_filtrado.groupby(["Sexo", "Nivel_Educativo"])["Probabilidad_Inscripcion"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=promedios, x="Sexo", y="Probabilidad_Inscripcion", hue="Nivel_Educativo", ax=ax)
    ax.set_title("Promedio de probabilidad por sexo y nivel educativo")
    ax.set_ylabel("Probabilidad de inscripción")
    ax.legend(title="Nivel Educativo", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("3️⃣ Relación entre sexo y sector económico")
    fig, ax = plt.subplots(figsize=(8, 4))
    tab = pd.crosstab(df_filtrado["Sector_Economico"], df_filtrado["Sexo"], normalize="index") * 100
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("%")
    ax.set_title("Distribución por sector")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📌 Distribuciones clave")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtrado, x="Edad", hue="Sexo", multiple="stack", ax=ax, bins=20)
        ax.set_title("Distribución de Edad por Sexo")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtrado, x="Nivel_Educativo", hue="Sexo", ax=ax)
        ax.set_title("Nivel Educativo por Sexo")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("🎯 Riesgo de Deserción")
    col1, col2 = st.columns([1, 2])
    with col1:
        riesgo_counts = df_filtrado["Riesgo_Desercion"].value_counts().reindex(["ALTO", "MEDIO", "BAJO"]).fillna(0)
        fig, ax = plt.subplots()
        riesgo_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=["red", "orange", "green"])
        ax.set_ylabel("")
        ax.set_title("Distribución de Riesgo")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=df_filtrado, x="Sector_Economico", y="Probabilidad_Inscripcion", hue="Riesgo_Desercion", ax=ax)
        ax.set_title("Probabilidad promedio por sector y riesgo")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.header("🤖 Predicción personalizada")
    with st.form("formulario"):
        edad = st.slider("Edad", 15, 90, 30)
        sexo_i = st.selectbox("Sexo", df["Sexo"].unique())
        nivel_i = st.selectbox("Nivel Educativo", df["Nivel_Educativo"].unique())
        estado_i = st.selectbox("Estado Civil", df["Estado_Civil"].unique())
        sector_i = st.selectbox("Sector Económico", df["Sector_Economico"].unique())
        submit = st.form_submit_button("Predecir")

        if submit:
            entrada = pd.DataFrame([{
                "Edad": edad,
                "Sexo": sexo_i,
                "Nivel_Educativo": nivel_i,
                "Estado_Civil": estado_i,
                "Sector_Economico": sector_i
            }])

            df_model = pd.get_dummies(df.drop(columns=[
                "Inscrito_Camara_Comercio", "Probabilidad_Inscripcion", "Prediccion_Modelo", "Riesgo_Desercion"
            ]), drop_first=True)
            X = df_model
            y = df["Inscrito_Camara_Comercio"]
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(X, y)

            entrada_dum = pd.get_dummies(entrada, drop_first=True)
            for col in X.columns:
                if col not in entrada_dum.columns:
                    entrada_dum[col] = 0
            entrada_dum = entrada_dum[X.columns]

            proba = modelo.predict_proba(entrada_dum)[0][1]
            riesgo = "ALTO ❌" if proba < 0.3 else "MEDIO ⚠️" if proba <= 0.6 else "BAJO ✅"

            st.success(f"📈 Probabilidad de inscripción: {proba:.2%}")
            st.info(f"🛑 Nivel de riesgo: {riesgo}")
