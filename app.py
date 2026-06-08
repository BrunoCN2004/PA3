import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Automatización e IA",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("Scopus.csv")

df = load_data()

st.title("Automatización, IA y Transformación del Trabajo")

st.markdown("""
Análisis bibliométrico basado en publicaciones indexadas en Scopus.
""")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Publicaciones", len(df))
col2.metric("Total citas", int(df["Cited by"].sum()))
col3.metric("Promedio citas", round(df["Cited by"].mean(),2))
col4.metric("Año más productivo", int(df["Year"].mode()[0]))

st.divider()

pubs_year = df.groupby("Year").size().reset_index(name="Publicaciones")

fig = px.bar(
    pubs_year,
    x="Year",
    y="Publicaciones",
    title="Publicaciones por Año"
)

st.plotly_chart(fig, use_container_width=True)
