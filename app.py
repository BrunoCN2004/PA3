import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(
    page_title="Dashboard IA y Automatización",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos custom ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuente y fondo general */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #6c63ff !important;
    }

    /* Tarjetas de métricas */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2b55 100%);
        border: 1px solid #6c63ff44;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(108,99,255,0.15);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(108,99,255,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #a0a0c0 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #43e097 !important;
    }

    /* Títulos */
    h1 { color: #ffffff !important; font-weight: 700 !important; }
    h2, h3 { color: #c9c6ff !important; font-weight: 600 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #1a1a2e;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #a0a0c0;
        font-weight: 500;
        padding: 8px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6c63ff, #a855f7) !important;
        color: #fff !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.4);
    }

    /* Fondo principal */
    .stApp { background-color: #0d0d1a; }

    /* Separadores */
    hr { border-color: #2a2a4a !important; }

    /* Búsqueda */
    .stTextInput input {
        background: #1a1a2e !important;
        border: 1px solid #6c63ff55 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Botón de descarga */
    .stDownloadButton button {
        background: linear-gradient(90deg, #6c63ff, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: opacity 0.2s !important;
    }
    .stDownloadButton button:hover { opacity: 0.85 !important; }

    /* Spinner */
    .stSpinner { color: #6c63ff !important; }
</style>
""", unsafe_allow_html=True)

# Paleta de colores para plotly
COLORS = {
    "purple": "#6c63ff",
    "violet": "#a855f7",
    "teal": "#43e097",
    "pink": "#f472b6",
    "amber": "#fbbf24",
    "blue": "#60a5fa",
    "bg": "#1a1a2e",
    "bg2": "#16213e",
    "text": "#e0e0f0",
    "grid": "#2a2a4a",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg2"],
        font=dict(family="Inter", color=COLORS["text"]),
        title_font=dict(size=16, color="#c9c6ff"),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#2d2b55", font_size=13, bordercolor="#6c63ff"),
    )
)


@st.cache_data
def load_data():
    return pd.read_csv("Scopus.csv")


df_raw = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Filtros")
    st.markdown("---")
    df = df_raw.copy()

    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique())
        selected_years = st.multiselect("📅 Años", years, default=years)
        df = df[df["Year"].isin(selected_years)]

    if "Document Type" in df.columns:
        docs = sorted(df["Document Type"].dropna().unique())
        selected_docs = st.multiselect("📄 Tipo de documento", docs, default=docs)
        df = df[df["Document Type"].isin(selected_docs)]

    if "Cited by" in df.columns:
        max_cit = int(df["Cited by"].max())
        min_cit = st.slider("📌 Citas mínimas", 0, max_cit, 0)
        df = df[df["Cited by"] >= min_cit]

    st.markdown("---")
    total = len(df)
    total_raw = len(df_raw)
    pct = round(100 * total / total_raw, 1) if total_raw else 0
    st.markdown(f"**{total}** de **{total_raw}** docs ({pct}%)")
    st.progress(pct / 100)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
    <h1 style='font-size:2.4rem; background: linear-gradient(90deg,#6c63ff,#a855f7,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🤖 IA, Automatización y el Futuro del Trabajo
    </h1>
    <p style='color:#a0a0c0; font-size:1rem; margin-top:-10px;'>
    Análisis bibliométrico · Fuente: Scopus
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Resumen", "📈 Tendencias", "👩‍🔬 Autores", "☁️ Abstracts", "🔍 Explorador"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Resumen
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    pubs = len(df)
    cites_total = int(df["Cited by"].sum()) if "Cited by" in df.columns else 0
    unique_authors = len(set(";".join(df["Authors"].fillna("")).split(";"))) if "Authors" in df.columns else 0
    journals = df["Source title"].nunique() if "Source title" in df.columns else 0
    avg_cites = round(cites_total / pubs, 1) if pubs else 0
    h_index_approx = 0
    if "Cited by" in df.columns:
        sorted_cites = sorted(df["Cited by"].dropna().astype(int), reverse=True)
        for i, c in enumerate(sorted_cites, 1):
            if c >= i:
                h_index_approx = i

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📚 Publicaciones", f"{pubs:,}")
    c2.metric("🔢 Total citas", f"{cites_total:,}")
    c3.metric("👤 Autores únicos", f"{unique_authors:,}")
    c4.metric("📰 Revistas", f"{journals:,}")
    c5.metric("📊 Citas / doc", f"{avg_cites}")
    c6.metric("🏅 h-index aprox.", f"{h_index_approx}")

    st.markdown("---")

    # Mini charts de resumen en dos columnas
    col_l, col_r = st.columns(2)

    with col_l:
        if "Year" in df.columns:
            pubs_year = df.groupby("Year").size().reset_index(name="n")
            fig = go.Figure(go.Bar(
                x=pubs_year["Year"], y=pubs_year["n"],
                marker=dict(
                    color=pubs_year["n"],
                    colorscale=[[0, "#6c63ff"], [1, "#f472b6"]],
                    showscale=False,
                ),
                hovertemplate="<b>%{x}</b><br>%{y} publicaciones<extra></extra>"
            ))
            fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                              title="Publicaciones por año", height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "Document Type" in df.columns:
            dtype = df["Document Type"].value_counts().head(6).reset_index()
            dtype.columns = ["Tipo", "n"]
            fig = px.pie(dtype, names="Tipo", values="n",
                         color_discrete_sequence=["#6c63ff", "#a855f7", "#f472b6",
                                                  "#43e097", "#fbbf24", "#60a5fa"],
                         hole=0.5)
            fig.update_traces(textposition="outside",
                              hovertemplate="<b>%{label}</b><br>%{value} docs (%{percent})<extra></extra>")
            fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                              title="Tipos de documento", height=280,
                              showlegend=True,
                              legend=dict(orientation="v", font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

    # Top revistas
    if "Source title" in df.columns and "Cited by" in df.columns:
        st.markdown("#### 🏆 Top 10 revistas por impacto")
        top_journals = (
            df.groupby("Source title")
            .agg(Publicaciones=("Source title", "count"), Citas=("Cited by", "sum"))
            .sort_values("Citas", ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.scatter(
            top_journals, x="Publicaciones", y="Citas",
            size="Citas", color="Citas",
            color_continuous_scale=["#6c63ff", "#f472b6"],
            hover_name="Source title",
            labels={"Source title": "Revista"},
            size_max=50,
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Tendencias
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if "Year" in df.columns:
        pubs_year = df.groupby("Year").size().reset_index(name="Publicaciones")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pubs_year["Year"], y=pubs_year["Publicaciones"],
            mode="lines+markers",
            line=dict(color=COLORS["purple"], width=3),
            marker=dict(size=8, color=COLORS["violet"],
                        line=dict(color="white", width=2)),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.12)",
            hovertemplate="<b>%{x}</b><br>%{y} publicaciones<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                          title="📈 Evolución de publicaciones", height=300)
        st.plotly_chart(fig, use_container_width=True)

        if "Cited by" in df.columns:
            cites_year = df.groupby("Year")["Cited by"].agg(["sum", "mean"]).reset_index()
            cites_year.columns = ["Year", "Total", "Promedio"]

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(
                x=cites_year["Year"], y=cites_year["Total"],
                name="Citas totales",
                marker_color=COLORS["violet"],
                opacity=0.85,
                hovertemplate="%{y:,.0f} citas totales<extra></extra>"
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=cites_year["Year"], y=cites_year["Promedio"].round(1),
                name="Promedio citas",
                line=dict(color=COLORS["teal"], width=2, dash="dot"),
                marker=dict(size=6),
                hovertemplate="%{y:.1f} citas promedio<extra></extra>"
            ), secondary_y=True)
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"],
                               title="📊 Citas totales y promedio por año",
                               height=320,
                               legend=dict(orientation="h", y=-0.25))
            fig2.update_yaxes(title_text="Citas totales", secondary_y=False,
                              gridcolor=COLORS["grid"])
            fig2.update_yaxes(title_text="Promedio / doc", secondary_y=True,
                              showgrid=False)
            st.plotly_chart(fig2, use_container_width=True)

    if "Abstract" in df.columns:
        text = " ".join(df["Abstract"].fillna("").astype(str)).lower()
        techs = [
            "artificial intelligence", "automation", "robotics",
            "machine learning", "deep learning", "digitalization",
            "industry 4.0", "future of work", "natural language processing",
            "neural network", "big data", "cloud computing",
        ]
        tech_df = pd.DataFrame(
            [{"Tecnología": t.title(), "Frecuencia": text.count(t)} for t in techs]
        ).sort_values("Frecuencia", ascending=False)

        fig3 = px.bar(
            tech_df, x="Frecuencia", y="Tecnología", orientation="h",
            color="Frecuencia",
            color_continuous_scale=["#6c63ff", "#a855f7", "#f472b6"],
            text="Frecuencia",
        )
        fig3.update_traces(textposition="outside",
                           hovertemplate="<b>%{y}</b><br>%{x} menciones<extra></extra>")
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"],
                           title="🔬 Tecnologías más mencionadas en abstracts",
                           height=420, coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending",
                                      gridcolor=COLORS["grid"]))
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Autores
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if "Authors" in df.columns and "Cited by" in df.columns:
        col_n, _ = st.columns([1, 3])
        with col_n:
            n_authors = st.selectbox("Mostrar top N autores", [10, 20, 30, 50], index=1)

        # Explotar autores
        rows = []
        for _, row in df.iterrows():
            for a in str(row["Authors"]).split(";"):
                a = a.strip()
                if a and a != "nan":
                    rows.append({
                        "Autor": a,
                        "Citas": row["Cited by"],
                        "Year": row.get("Year", None)
                    })
        adf = pd.DataFrame(rows)
        top_authors = (
            adf.groupby("Autor")
            .agg(Citas=("Citas", "sum"), Publicaciones=("Autor", "count"))
            .sort_values("Citas", ascending=False)
            .head(n_authors)
            .reset_index()
        )

        fig = go.Figure(go.Bar(
            x=top_authors["Citas"],
            y=top_authors["Autor"],
            orientation="h",
            marker=dict(
                color=top_authors["Citas"],
                colorscale=[[0, "#6c63ff"], [0.5, "#a855f7"], [1, "#f472b6"]],
                showscale=False,
            ),
            customdata=top_authors["Publicaciones"],
            hovertemplate="<b>%{y}</b><br>Citas: %{x:,}<br>Publicaciones: %{customdata}<extra></extra>"
        ))
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title=f"🏅 Top {n_authors} autores más citados",
            height=max(400, n_authors * 26),
            yaxis=dict(categoryorder="total ascending", gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter productividad vs impacto
        st.markdown("#### 🎯 Productividad vs Impacto")
        top_scatter = (
            adf.groupby("Autor")
            .agg(Citas=("Citas", "sum"), Publicaciones=("Autor", "count"))
            .reset_index()
        )
        top_scatter = top_scatter[top_scatter["Publicaciones"] >= 2]
        fig2 = px.scatter(
            top_scatter, x="Publicaciones", y="Citas",
            size="Citas", color="Citas",
            color_continuous_scale=["#6c63ff", "#f472b6"],
            hover_name="Autor", size_max=40,
            labels={"Publicaciones": "N° de publicaciones",
                    "Citas": "Total de citas"},
        )
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Abstracts / Wordcloud
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    if "Abstract" in df.columns:
        col_wc, col_freq = st.columns([3, 2])

        with col_wc:
            st.markdown("#### ☁️ Nube de palabras")
            stopwords_extra = {
                "this", "that", "with", "from", "which", "have", "been",
                "also", "paper", "study", "research", "using", "based",
                "results", "proposed", "used", "can", "are", "the", "and",
                "for", "not", "were", "has", "its", "more", "their",
            }
            text_full = " ".join(df["Abstract"].fillna("").astype(str))
            wc = WordCloud(
                width=900, height=450,
                background_color="#1a1a2e",
                colormap="cool",
                stopwords=stopwords_extra,
                max_words=150,
                contour_width=0,
                prefer_horizontal=0.8,
            ).generate(text_full)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            fig_wc.patch.set_facecolor("#1a1a2e")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

        with col_freq:
            st.markdown("#### 🔤 Términos frecuentes")
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text_full.lower())
            bad = {"this", "that", "with", "from", "which", "have", "been",
                   "also", "paper", "study", "research", "using", "based",
                   "results", "proposed", "used", "these", "their", "more",
                   "through", "such", "about", "they", "into", "other",
                   "approach", "show", "data", "work", "both", "than",
                   "methods", "model", "models", "method"}
            freq = Counter(w for w in words if w not in bad)
            top_words = pd.DataFrame(freq.most_common(20), columns=["Término", "Frec"])

            fig3 = go.Figure(go.Bar(
                x=top_words["Frec"], y=top_words["Término"],
                orientation="h",
                marker=dict(color=top_words["Frec"],
                            colorscale=[[0, "#6c63ff"], [1, "#43e097"]],
                            showscale=False),
                hovertemplate="%{y}: %{x} ocurrencias<extra></extra>"
            ))
            fig3.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=480,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(categoryorder="total ascending",
                           gridcolor=COLORS["grid"]),
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No hay columna 'Abstract' en el dataset.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – Explorador
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    col_s, col_sort = st.columns([3, 1])
    with col_s:
        search = st.text_input("🔍 Buscar por título o abstract", placeholder="ej: deep learning, automation…")
    with col_sort:
        sort_col = st.selectbox(
            "Ordenar por",
            [c for c in ["Cited by", "Year"] if c in df.columns],
            index=0
        )

    filtered = df.copy()

    if search:
        mask = pd.Series([False] * len(filtered), index=filtered.index)
        for col in ["Title", "Abstract"]:
            if col in filtered.columns:
                mask |= filtered[col].str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=False)

    # Mostrar nro de resultados
    st.markdown(f"**{len(filtered)}** documentos encontrados")

    # Selección de columnas a mostrar
    display_cols = [c for c in ["Title", "Authors", "Year", "Source title",
                                "Cited by", "Document Type"] if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    st.markdown("---")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            label="⬇️ Descargar CSV filtrado",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="datos_filtrados.csv",
            mime="text/csv",
        )
