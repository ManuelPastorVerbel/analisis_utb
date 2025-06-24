
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análisis de Intención Emprendedora", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv("modelo_emprendimiento_resultados.csv")

df = cargar_datos()

st.title("📊 Análisis de Intención Emprendedora - Barranquilla 2025")

st.markdown("""
Este panel responde a las preguntas del proyecto integrador relacionadas con la intención emprendedora de los ciudadanos que participan en la ruta de desarrollo productivo.
Seleccione filtros para explorar el comportamiento por variables clave como sexo, edad, sector y nivel educativo.
""")

# Filtros laterales
st.sidebar.header("🎛️ Filtros de Exploración")
sexo = st.sidebar.multiselect("Sexo", df["Sexo"].unique(), default=df["Sexo"].unique())
sector = st.sidebar.multiselect("Sector Económico", df["Sector_Economico"].unique(), default=df["Sector_Economico"].unique())
nivel = st.sidebar.multiselect("Nivel Educativo", df["Nivel_Educativo"].unique(), default=df["Nivel_Educativo"].unique())

df_filtrado = df[
    (df["Sexo"].isin(sexo)) &
    (df["Sector_Economico"].isin(sector)) &
    (df["Nivel_Educativo"].isin(nivel))
]

# 1. Perfil socioeconómico con intención emprendedora
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

# 2. Diferencia entre mujeres y hombres con intención emprendedora
st.subheader("2️⃣ Diferencias entre mujeres y hombres con intención emprendedora")

# Agrupamos y sacamos promedio 
# Normalizar nombres de nivel educativo
df_filtrado["Nivel_Educativo"] = df_filtrado["Nivel_Educativo"].str.upper().str.strip()

# Agrupamos y sacamos promedio
promedios = df_filtrado.groupby(["Sexo", "Nivel_Educativo"])["Probabilidad_Inscripcion"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=promedios, x="Sexo", y="Probabilidad_Inscripcion", hue="Nivel_Educativo", ax=ax)
ax.set_title("Promedio de probabilidad por sexo y nivel educativo")
ax.set_ylabel("Probabilidad de inscripción")
ax.legend(title="Nivel Educativo", bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)


# 3. ¿El sexo influye en el tipo de sector?
st.subheader("3️⃣ ¿Existe relación entre el sexo y el sector económico de emprendimiento?")

fig, ax = plt.subplots(figsize=(8,4))
sexo_sector = pd.crosstab(df_filtrado["Sector_Economico"], df_filtrado["Sexo"], normalize="index") * 100
sexo_sector.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
ax.set_ylabel("%")
ax.set_title("Distribución de sexo por sector económico")
st.pyplot(fig)

# 4. ¿La edad se relaciona con el sexo en el emprendimiento?
st.subheader("4️⃣ ¿La edad se relaciona con el sexo en el emprendimiento?")

fig, ax = plt.subplots()
sns.histplot(data=df_filtrado, x="Edad", hue="Sexo", multiple="stack", bins=20, ax=ax)
ax.set_title("Distribución de edad por sexo")
st.pyplot(fig)

st.markdown("📌 Datos fuente: Ruta de Desarrollo Productivo - Centro de Oportunidades Barranquilla (2025)")
