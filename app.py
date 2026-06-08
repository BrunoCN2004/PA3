
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="Dashboard IA y Automatización", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Scopus.csv")

df = load_data()

st.sidebar.header("Filtros Globales")

if "Year" in df.columns:
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.sidebar.multiselect("Años", years, default=years)
    df = df[df["Year"].isin(selected_years)]

if "Document Type" in df.columns:
    docs = sorted(df["Document Type"].dropna().unique())
    selected_docs = st.sidebar.multiselect("Tipo de documento", docs, default=docs)
    df = df[df["Document Type"].isin(selected_docs)]

if "Cited by" in df.columns:
    min_cit = st.sidebar.slider("Citas mínimas", 0, int(df["Cited by"].max()), 0)
    df = df[df["Cited by"] >= min_cit]

st.title("🤖 Automatización, IA y Transformación del Trabajo")
st.markdown("Dashboard bibliométrico basado en Scopus")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Resumen", "Tendencias", "Autores", "Abstracts", "Explorador"]
)

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Publicaciones", len(df))
    c2.metric("Total citas", int(df["Cited by"].sum()) if "Cited by" in df.columns else 0)
    c3.metric("Autores únicos", len(set(";".join(df["Authors"].fillna("")).split(";"))) if "Authors" in df.columns else 0)
    c4.metric("Revistas", df["Source title"].nunique() if "Source title" in df.columns else 0)

with tab2:
    if "Year" in df.columns:
        pubs = df.groupby("Year").size().reset_index(name="Publicaciones")
        st.plotly_chart(px.line(pubs,x="Year",y="Publicaciones",markers=True,title="Publicaciones por Año"), use_container_width=True)

        if "Cited by" in df.columns:
            cites = df.groupby("Year")["Cited by"].sum().reset_index()
            st.plotly_chart(px.bar(cites,x="Year",y="Cited by",title="Citas por Año"), use_container_width=True)

    if "Abstract" in df.columns:
        text = " ".join(df["Abstract"].fillna("").astype(str)).lower()
        techs = [
            "artificial intelligence","automation","robotics",
            "machine learning","deep learning","digitalization",
            "industry 4.0","future of work"
        ]
        counts = [{"Tecnología":t,"Frecuencia":text.count(t)} for t in techs]
        tech_df = pd.DataFrame(counts)
        st.plotly_chart(px.treemap(tech_df,path=["Tecnología"],values="Frecuencia",
                                   title="Tecnologías más mencionadas"),
                                   use_container_width=True)

with tab3:
    if "Authors" in df.columns and "Cited by" in df.columns:
        authors=[]
        for _,row in df.iterrows():
            for a in str(row["Authors"]).split(";"):
                authors.append([a.strip(),row["Cited by"]])
        adf = pd.DataFrame(authors,columns=["Autor","Citas"])
        top = adf.groupby("Autor")["Citas"].sum().sort_values(ascending=False).head(20).reset_index()
        st.plotly_chart(px.bar(top,x="Citas",y="Autor",orientation="h",
                               title="Autores más citados"),
                               use_container_width=True)

with tab4:
    if "Abstract" in df.columns:
        text = " ".join(df["Abstract"].fillna("").astype(str))
        wc = WordCloud(width=1200,height=500,background_color="white").generate(text)
        fig,ax = plt.subplots(figsize=(12,5))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

with tab5:
    search = st.text_input("Buscar por título")
    if search and "Title" in df.columns:
        st.dataframe(df[df["Title"].str.contains(search,case=False,na=False)])
    else:
        st.dataframe(df)

    st.download_button(
        "Descargar CSV filtrado",
        df.to_csv(index=False),
        "datos_filtrados.csv",
        "text/csv"
    )
