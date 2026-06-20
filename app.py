# -*- coding: utf-8 -*-
"""
Amazon Customer Review Analytics
CRM & Sentiment Analysis Dashboard - Pastel Edition
Final Project - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re
import time
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Amazon Customer Review Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PASTEL COLOR PALETTE
# ============================================================
PASTEL_PINK = "#F7C8D9"
PASTEL_PINK_DARK = "#E8A1BF"
PASTEL_PURPLE = "#D8C7F0"
PASTEL_PURPLE_DARK = "#B79CE8"
PASTEL_BLUE = "#BFE0F2"
PASTEL_BLUE_DARK = "#8FC6E8"
PASTEL_LAVENDER = "#E3D5F1"
TEXT_DARK = "#4A3F55"
TEXT_MUTED = "#7C6E8A"
BG_SOFT = "#FBF8FD"
CARD_BG = "#FFFFFF"

# Sentiment colors (soft/pastel but distinguishable, requested green/orange/red kept soft)
COLOR_POSITIVE = "#A8D8B9"   # pastel green
COLOR_NEUTRAL = "#F7D8A8"    # pastel orange
COLOR_NEGATIVE = "#F2A8A8"   # pastel red

SENTIMENT_COLOR_MAP = {
    "Positive": COLOR_POSITIVE,
    "Neutral": COLOR_NEUTRAL,
    "Negative": COLOR_NEGATIVE,
}

PASTEL_SEQUENTIAL = [PASTEL_PINK, PASTEL_PURPLE, PASTEL_BLUE, PASTEL_LAVENDER,
                      PASTEL_PINK_DARK, PASTEL_PURPLE_DARK, PASTEL_BLUE_DARK, "#CDEAD8"]

PLOTLY_TEMPLATE = "plotly_white"

# ============================================================
# CUSTOM CSS
# ============================================================
def load_css(dark_mode=False):
    if dark_mode:
        bg = "#211B2B"
        card_bg = "#2C2438"
        text = "#F1E9F7"
        muted = "#C9B8DC"
        sidebar_bg = "linear-gradient(180deg, #2C2438 0%, #211B2B 100%)"
        header_grad = "linear-gradient(120deg, #6B5B95 0%, #C68FB0 50%, #8FB4D9 100%)"
    else:
        bg = BG_SOFT
        card_bg = CARD_BG
        text = TEXT_DARK
        muted = TEXT_MUTED
        sidebar_bg = f"linear-gradient(180deg, {PASTEL_LAVENDER} 0%, {PASTEL_PINK} 100%)"
        header_grad = f"linear-gradient(120deg, {PASTEL_PINK_DARK} 0%, {PASTEL_PURPLE_DARK} 50%, {PASTEL_BLUE_DARK} 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        color: {text};
    }}

    .stApp {{
        background-color: {bg};
    }}

    /* Header */
    .main-header {{
        background: {header_grad};
        padding: 28px 36px;
        border-radius: 22px;
        margin-bottom: 26px;
        box-shadow: 0 8px 24px rgba(150, 120, 180, 0.25);
    }}
    .main-header h1 {{
        color: #ffffff;
        font-weight: 800;
        font-size: 2.1rem;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .main-header p {{
        color: #ffffffdd;
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        font-weight: 500;
    }}

    /* KPI Cards */
    .kpi-card {{
        background: {card_bg};
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 6px 18px rgba(150, 120, 180, 0.14);
        border: 1px solid rgba(180,150,200,0.18);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }}
    .kpi-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 14px 28px rgba(150, 120, 180, 0.28);
    }}
    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {muted};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.7rem;
        font-weight: 800;
        color: {text};
    }}
    .kpi-icon {{
        font-size: 1.4rem;
        margin-bottom: 6px;
    }}

    /* Section card wrapper - st.container(border=True) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {card_bg};
        border-radius: 20px !important;
        box-shadow: 0 6px 18px rgba(150,120,180,0.12);
        margin-bottom: 22px;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-radius: 20px !important;
        border: 1px solid rgba(180,150,200,0.14) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{
        gap: 0.6rem;
    }}

    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 14px;
        border-left: 5px solid {PASTEL_PURPLE_DARK};
        padding-left: 10px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid rgba(180,150,200,0.2);
    }}
    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div p {{
        font-weight: 600 !important;
        color: {text} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {{
        color: {muted} !important;
        opacity: 0.9 !important;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {{
        color: {text} !important;
    }}

    /* Buttons */
    .stButton button, .stDownloadButton button {{
        background: linear-gradient(120deg, {PASTEL_PURPLE_DARK}, {PASTEL_PINK_DARK});
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.55em 1.4em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(150,120,180,0.25);
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(150,120,180,0.35);
        filter: brightness(1.05);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PASTEL_LAVENDER}55;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        padding: 8px 16px;
    }}

    /* Badge / pill */
    .pill {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 8px;
    }}

    /* Dataframe rounding */
    [data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
    }}

    /* Hide default footer/menu for clean look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {PASTEL_PURPLE_DARK}55, transparent);
        margin: 18px 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(col, icon, label, value, color=PASTEL_PURPLE_DARK):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
DEFAULT_FILE = "Reviews_Cleaned_Streamlit.csv"

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    # Ensure expected columns exist; fill safe defaults if missing
    if "Sentiment_Label" not in df.columns and "Score" in df.columns:
        def lbl(s):
            if s >= 4:
                return "Positive"
            elif s == 3:
                return "Neutral"
            else:
                return "Negative"
        df["Sentiment_Label"] = df["Score"].apply(lbl)

    if "Cleaned_Review" not in df.columns:
        for c in ["Review", "Text", "Summary"]:
            if c in df.columns:
                df["Cleaned_Review"] = df[c].astype(str)
                break

    if "Review_Length" not in df.columns and "Cleaned_Review" in df.columns:
        df["Review_Length"] = df["Cleaned_Review"].astype(str).apply(len)

    if "Helpful_Ratio" not in df.columns and {"HelpfulnessNumerator","HelpfulnessDenominator"}.issubset(df.columns):
        df["Helpful_Ratio"] = df.apply(
            lambda r: r["HelpfulnessNumerator"]/r["HelpfulnessDenominator"]
            if r["HelpfulnessDenominator"] > 0 else 0, axis=1
        )

    df["Sentiment_Label"] = df["Sentiment_Label"].astype(str).str.capitalize()
    return df


def get_data():
    try:
        df = load_data(DEFAULT_FILE)
        st.session_state["data_source"] = f"Loaded `{DEFAULT_FILE}` from working directory."
        return df
    except FileNotFoundError:
        uploaded = st.sidebar.file_uploader(
            "📁 Upload Reviews_Cleaned_Streamlit.csv", type=["csv"]
        )
        if uploaded is not None:
            df = load_data(uploaded)
            st.session_state["data_source"] = "Loaded from uploaded file."
            return df
        else:
            st.warning(
                f"⚠️ File **{DEFAULT_FILE}** tidak ditemukan di folder kerja. "
                "Silakan upload file CSV melalui sidebar untuk menjalankan dashboard."
            )
            st.stop()


@st.cache_data(show_spinner=False)
def get_top_words(text_series, n=20, stopwords_extra=None):
    sw = set(STOPWORDS)
    if stopwords_extra:
        sw.update(stopwords_extra)
    words = []
    for txt in text_series.dropna().astype(str):
        tokens = re.findall(r"[a-zA-Z]{3,}", txt.lower())
        words.extend([t for t in tokens if t not in sw])
    counter = Counter(words)
    return counter.most_common(n)


def make_wordcloud_fig(text_series, colormap, bg="#FFFFFF"):
    text = " ".join(text_series.dropna().astype(str).tolist())
    if not text.strip():
        text = "no_data_available"
    wc = WordCloud(
        width=500, height=350, background_color=bg,
        colormap=colormap, stopwords=set(STOPWORDS),
        max_words=120, prefer_horizontal=0.9
    ).generate(text)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    return fig


def style_plotly(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family="Poppins, sans-serif", color=TEXT_DARK, size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Poppins"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# SESSION STATE
# ============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Review Analytics")
    st.markdown("##### CRM & Sentiment Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        [
            "🏠 Overview Dashboard",
            "😊 Sentiment Analysis",
            "⭐ Rating Analysis",
            "📦 Product Analysis",
            "👤 Customer Analysis",
            "☁️ WordCloud Analysis",
            "📝 Review Explorer",
            "📈 CRM Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)
    st.markdown("---")

load_css(dark_mode=st.session_state.dark_mode)

if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📊 Amazon Customer Review Analytics</h1>
    <p>CRM Intelligence & Sentiment Analysis — Final Project Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Memuat data review..."):
    df_raw = get_data()

# ============================================================
# GLOBAL FILTERS (in sidebar, below nav for clarity)
# ============================================================
with st.sidebar:
    st.markdown("### 🔍 Filter Global")

    sentiments_all = sorted(df_raw["Sentiment_Label"].dropna().unique().tolist())
    sel_sentiment = st.multiselect("Sentiment", sentiments_all, default=sentiments_all)

    scores_all = sorted(df_raw["Score"].dropna().unique().tolist()) if "Score" in df_raw.columns else []
    sel_score = st.multiselect("Score (Rating)", scores_all, default=scores_all)

    if "Review_Year" in df_raw.columns:
        years_all = sorted(df_raw["Review_Year"].dropna().unique().tolist())
        sel_year = st.multiselect("Year", years_all, default=years_all)
    else:
        sel_year = None

    if "Review_Month" in df_raw.columns:
        months_all = sorted(df_raw["Review_Month"].dropna().unique().tolist())
        sel_month = st.multiselect("Month", months_all, default=months_all)
    else:
        sel_month = None

    if "ProductId" in df_raw.columns:
        products_all = sorted(df_raw["ProductId"].dropna().unique().tolist())
        sel_product = st.multiselect(
            "Product (kosongkan = semua)", products_all, default=[]
        )
    else:
        sel_product = []

    st.markdown("---")
    st.caption(st.session_state.get("data_source", ""))
    st.caption(f"Total baris dataset: **{len(df_raw):,}**")

# Apply filters
df = df_raw.copy()
if sel_sentiment:
    df = df[df["Sentiment_Label"].isin(sel_sentiment)]
if sel_score:
    df = df[df["Score"].isin(sel_score)]
if sel_year:
    df = df[df["Review_Year"].isin(sel_year)]
if sel_month:
    df = df[df["Review_Month"].isin(sel_month)]
if sel_product:
    df = df[df["ProductId"].isin(sel_product)]

if df.empty:
    st.error("🚫 Tidak ada data yang sesuai dengan filter. Silakan ubah filter di sidebar.")
    st.stop()


# ============================================================
# PAGE: OVERVIEW DASHBOARD
# ============================================================
if page == "🏠 Overview Dashboard":

    total_reviews = len(df)
    total_products = df["ProductId"].nunique() if "ProductId" in df.columns else 0
    total_customers = df["UserId"].nunique() if "UserId" in df.columns else 0
    avg_rating = df["Score"].mean() if "Score" in df.columns else 0
    avg_helpful = df["Helpful_Ratio"].mean() if "Helpful_Ratio" in df.columns else 0
    avg_length = df["Review_Length"].mean() if "Review_Length" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "📝", "Total Reviews", f"{total_reviews:,}", PASTEL_PURPLE_DARK)
    kpi_card(c2, "📦", "Total Products", f"{total_products:,}", PASTEL_PINK_DARK)
    kpi_card(c3, "👤", "Total Customers", f"{total_customers:,}", PASTEL_BLUE_DARK)

    st.write("")
    c4, c5, c6 = st.columns(3)
    kpi_card(c4, "⭐", "Average Rating", f"{avg_rating:.2f} / 5", PASTEL_PURPLE_DARK)
    kpi_card(c5, "👍", "Avg Helpful Ratio", f"{avg_helpful*100:.1f}%", PASTEL_PINK_DARK)
    kpi_card(c6, "📏", "Avg Review Length", f"{avg_length:.0f} chars", PASTEL_BLUE_DARK)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("😊 Sentiment Distribution")
            sent_counts = df["Sentiment_Label"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                sent_counts, names="Sentiment", values="Count", hole=0.5,
                color="Sentiment", color_discrete_map=SENTIMENT_COLOR_MAP,
            )
            fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} (%{percent})")
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("⭐ Rating Distribution")
            rating_counts = df["Score"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["Score", "Count"]
            fig = px.bar(
                rating_counts, x="Score", y="Count", text="Count",
                color="Score", color_continuous_scale=[PASTEL_BLUE, PASTEL_PURPLE_DARK],
            )
            fig.update_traces(textposition="outside", hovertemplate="Rating %{x}★: %{y} reviews")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("📈 Reviews Over Time (Yearly)")
            if "Review_Year" in df.columns:
                yearly = df.groupby("Review_Year").size().reset_index(name="Count")
                fig = px.line(
                    yearly, x="Review_Year", y="Count", markers=True,
                    color_discrete_sequence=[PASTEL_PURPLE_DARK],
                )
                fig.update_traces(line=dict(width=4), marker=dict(size=9, color=PASTEL_PINK_DARK),
                                   hovertemplate="Year %{x}: %{y} reviews")
                st.plotly_chart(style_plotly(fig, 340), use_container_width=True)
            else:
                st.info("Kolom Review_Year tidak tersedia.")

    with col4:
        with st.container(border=True):
            section_header("🗓️ Monthly Trend")
            if "Review_Month" in df.columns:
                monthly = df.groupby("Review_Month").size().reset_index(name="Count").sort_values("Review_Month")
                fig = px.area(
                    monthly, x="Review_Month", y="Count",
                    color_discrete_sequence=[PASTEL_BLUE_DARK],
                )
                fig.update_traces(hovertemplate="Bulan %{x}: %{y} reviews")
                st.plotly_chart(style_plotly(fig, 340), use_container_width=True)
            else:
                st.info("Kolom Review_Month tidak tersedia.")


# ============================================================
# PAGE: SENTIMENT ANALYSIS
# ============================================================
elif page == "😊 Sentiment Analysis":

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("😊 Sentiment Distribution")
            sent_counts = df["Sentiment_Label"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(sent_counts, names="Sentiment", values="Count", hole=0.45,
                          color="Sentiment", color_discrete_map=SENTIMENT_COLOR_MAP)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("🔢 Sentiment Count")
            fig = px.bar(sent_counts, x="Sentiment", y="Count", text="Count",
                          color="Sentiment", color_discrete_map=SENTIMENT_COLOR_MAP)
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("⭐ Average Rating per Sentiment")
            avg_rate = df.groupby("Sentiment_Label")["Score"].mean().reset_index()
            fig = px.bar(avg_rate, x="Sentiment_Label", y="Score", text_auto=".2f",
                          color="Sentiment_Label", color_discrete_map=SENTIMENT_COLOR_MAP)
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)

    with col4:
        with st.container(border=True):
            section_header("📏 Review Length by Sentiment")
            fig = px.box(df, x="Sentiment_Label", y="Review_Length",
                          color="Sentiment_Label", color_discrete_map=SENTIMENT_COLOR_MAP)
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)

    with st.container(border=True):
        section_header("👍 Helpful Ratio by Sentiment")
        fig = px.box(df, x="Sentiment_Label", y="Helpful_Ratio",
                      color="Sentiment_Label", color_discrete_map=SENTIMENT_COLOR_MAP)
        st.plotly_chart(style_plotly(fig, 380), use_container_width=True)


# ============================================================
# PAGE: RATING ANALYSIS
# ============================================================
elif page == "⭐ Rating Analysis":

    with st.container(border=True):
        section_header("⭐ Distribusi Rating (1–5 Bintang)")
        rating_counts = df["Score"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Score", "Count"]
        fig = px.bar(rating_counts, x="Score", y="Count", text="Count",
                      color="Score", color_continuous_scale=[PASTEL_BLUE, PASTEL_PURPLE_DARK, PASTEL_PINK_DARK])
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("📊 Rating vs Review Length")
            sample = df.sample(min(3000, len(df)), random_state=42)
            fig = px.scatter(sample, x="Score", y="Review_Length", color="Sentiment_Label",
                              color_discrete_map=SENTIMENT_COLOR_MAP, opacity=0.6)
            st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("📊 Rating vs Helpful Ratio")
            fig = px.scatter(sample, x="Score", y="Helpful_Ratio", color="Sentiment_Label",
                              color_discrete_map=SENTIMENT_COLOR_MAP, opacity=0.6)
            st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    with st.container(border=True):
        section_header("📏 Average Review Length per Rating")
        avg_len = df.groupby("Score")["Review_Length"].mean().reset_index()
        fig = px.bar(avg_len, x="Score", y="Review_Length", text_auto=".0f",
                      color="Score", color_continuous_scale=[PASTEL_PINK, PASTEL_PURPLE_DARK])
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_plotly(fig, 360), use_container_width=True)


# ============================================================
# PAGE: PRODUCT ANALYSIS
# ============================================================
elif page == "📦 Product Analysis":

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("🔝 Top 10 Most Reviewed Products")
            top_reviewed = df["ProductId"].value_counts().head(10).reset_index()
            top_reviewed.columns = ["ProductId", "Review Count"]
            fig = px.bar(top_reviewed, x="Review Count", y="ProductId", orientation="h",
                          text="Review Count", color="Review Count",
                          color_continuous_scale=[PASTEL_BLUE, PASTEL_PURPLE_DARK])
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_plotly(fig, 420), use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("🌟 Top 10 Highest Rated Products (min. 20 reviews)")
            prod_stats = df.groupby("ProductId")["Score"].agg(["mean", "count"]).reset_index()
            prod_stats = prod_stats[prod_stats["count"] > 20].sort_values("mean", ascending=False).head(10)
            if prod_stats.empty:
                st.info("Tidak ada produk dengan jumlah review > 20 pada filter saat ini.")
            else:
                fig = px.bar(prod_stats, x="mean", y="ProductId", orientation="h",
                              text=prod_stats["mean"].round(2), color="mean",
                              color_continuous_scale=[PASTEL_PINK, PASTEL_PURPLE_DARK])
                fig.update_traces(textposition="outside")
                fig.update_coloraxes(showscale=False)
                fig.update_yaxes(categoryorder="total ascending")
                fig.update_xaxes(title="Average Rating")
                st.plotly_chart(style_plotly(fig, 420), use_container_width=True)

    with st.container(border=True):
        section_header("📊 Product Sentiment Analysis (Top 10 by Volume)")
        top_ids = df["ProductId"].value_counts().head(10).index
        sub = df[df["ProductId"].isin(top_ids)]
        stacked = sub.groupby(["ProductId", "Sentiment_Label"]).size().reset_index(name="Count")
        fig = px.bar(stacked, x="ProductId", y="Count", color="Sentiment_Label",
                      color_discrete_map=SENTIMENT_COLOR_MAP, barmode="stack")
        st.plotly_chart(style_plotly(fig, 420), use_container_width=True)


# ============================================================
# PAGE: CUSTOMER ANALYSIS
# ============================================================
elif page == "👤 Customer Analysis":

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section_header("🏆 Top Active Customers")
            top_cust = df["UserId"].value_counts().head(10).reset_index()
            top_cust.columns = ["UserId", "Review Count"]
            fig = px.bar(top_cust, x="Review Count", y="UserId", orientation="h",
                          text="Review Count", color="Review Count",
                          color_continuous_scale=[PASTEL_PURPLE, PASTEL_PINK_DARK])
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_plotly(fig, 400), use_container_width=True)

    with col2:
        with st.container(border=True):
            section_header("📊 Customer Review Distribution")
            cust_counts = df["UserId"].value_counts()
            fig = px.histogram(cust_counts, nbins=30, color_discrete_sequence=[PASTEL_BLUE_DARK])
            fig.update_layout(xaxis_title="Jumlah Review per Customer", yaxis_title="Jumlah Customer", showlegend=False)
            st.plotly_chart(style_plotly(fig, 400), use_container_width=True)

    with st.container(border=True):
        section_header("👍 Top Helpful Reviewers (berdasarkan HelpfulnessNumerator)")
        if "HelpfulnessNumerator" in df.columns:
            top_helpful = df.groupby("UserId")["HelpfulnessNumerator"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_helpful, x="HelpfulnessNumerator", y="UserId", orientation="h",
                          text="HelpfulnessNumerator", color="HelpfulnessNumerator",
                          color_continuous_scale=[PASTEL_LAVENDER, PASTEL_PURPLE_DARK])
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_plotly(fig, 400), use_container_width=True)
        else:
            st.info("Kolom HelpfulnessNumerator tidak tersedia.")


# ============================================================
# PAGE: WORDCLOUD ANALYSIS
# ============================================================
elif page == "☁️ WordCloud Analysis":

    with st.container(border=True):
        section_header("☁️ WordCloud per Sentiment")

        colA, colB, colC = st.columns(3)
        cmaps = {"Positive": "Greens", "Neutral": "Oranges", "Negative": "Reds"}
        cols_map = {"Positive": colA, "Neutral": colB, "Negative": colC}

        text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "Text"

        for sentiment, col in cols_map.items():
            with col:
                st.markdown(f"**{sentiment} WordCloud**")
                sub_text = df[df["Sentiment_Label"] == sentiment][text_col]
                if len(sub_text) == 0:
                    st.info("Tidak ada data.")
                else:
                    with st.spinner(f"Membuat wordcloud {sentiment}..."):
                        fig = make_wordcloud_fig(sub_text, cmaps[sentiment])
                        st.pyplot(fig, use_container_width=True)

    with st.container(border=True):
        section_header("🔠 Top 20 Words per Sentiment")
        tabs = st.tabs(["😊 Positive", "😐 Neutral", "☹️ Negative"])
        bar_colors = {"Positive": COLOR_POSITIVE, "Neutral": COLOR_NEUTRAL, "Negative": COLOR_NEGATIVE}
        for tab, sentiment in zip(tabs, ["Positive", "Neutral", "Negative"]):
            with tab:
                sub_text = df[df["Sentiment_Label"] == sentiment][text_col]
                top_words = get_top_words(sub_text, n=20)
                if not top_words:
                    st.info("Tidak ada data.")
                else:
                    wdf = pd.DataFrame(top_words, columns=["Word", "Count"])
                    fig = px.bar(wdf.sort_values("Count"), x="Count", y="Word", orientation="h",
                                  text="Count", color_discrete_sequence=[bar_colors[sentiment]])
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(style_plotly(fig, 480), use_container_width=True)


# ============================================================
# PAGE: REVIEW EXPLORER
# ============================================================
elif page == "📝 Review Explorer":

    with st.container(border=True):
        section_header("🔎 Cari Review")
        search_term = st.text_input("Cari berdasarkan Summary / Text / Review", placeholder="contoh: delicious, terrible, packaging...")

        explorer_df = df.copy()
        if search_term:
            mask = pd.Series(False, index=explorer_df.index)
            for c in ["Summary", "Text", "Review"]:
                if c in explorer_df.columns:
                    mask = mask | explorer_df[c].astype(str).str.contains(search_term, case=False, na=False)
            explorer_df = explorer_df[mask]

        st.caption(f"Menampilkan **{len(explorer_df):,}** dari **{len(df):,}** review (setelah filter & pencarian)")

        display_cols = [c for c in ["ProductId", "Score", "Sentiment_Label", "Summary", "Text", "Helpful_Ratio"]
                         if c in explorer_df.columns]
        st.dataframe(
            explorer_df[display_cols],
            use_container_width=True,
            height=460,
            column_config={
                "Helpful_Ratio": st.column_config.ProgressColumn(
                    "Helpful Ratio", min_value=0, max_value=1, format="%.2f"
                ),
                "Score": st.column_config.NumberColumn("Score", format="⭐ %d"),
            },
        )

        csv_data = explorer_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_reviews.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE: CRM INSIGHTS
# ============================================================
elif page == "📈 CRM Insights":

    total_reviews = len(df)
    sent_pct = (df["Sentiment_Label"].value_counts(normalize=True) * 100).round(1)
    pos_pct = sent_pct.get("Positive", 0)
    neu_pct = sent_pct.get("Neutral", 0)
    neg_pct = sent_pct.get("Negative", 0)
    avg_rating = df["Score"].mean()

    # Most positive / negative / helpful product (min review threshold for stability)
    prod_group = df.groupby("ProductId").agg(
        avg_score=("Score", "mean"),
        review_count=("Score", "count"),
        avg_helpful=("Helpful_Ratio", "mean") if "Helpful_Ratio" in df.columns else ("Score", "count"),
    ).reset_index()
    prod_group_filtered = prod_group[prod_group["review_count"] >= 5]
    if prod_group_filtered.empty:
        prod_group_filtered = prod_group

    most_positive_product = prod_group_filtered.sort_values("avg_score", ascending=False).iloc[0]["ProductId"] \
        if not prod_group_filtered.empty else "N/A"
    most_negative_product = prod_group_filtered.sort_values("avg_score", ascending=True).iloc[0]["ProductId"] \
        if not prod_group_filtered.empty else "N/A"
    most_helpful_product = prod_group_filtered.sort_values("avg_helpful", ascending=False).iloc[0]["ProductId"] \
        if not prod_group_filtered.empty else "N/A"

    csi = (avg_rating / 5) * 100  # Customer Satisfaction Index

    with st.container(border=True):
        section_header("📋 Ringkasan Sentimen & Kepuasan Pelanggan")
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "😊", "Positive %", f"{pos_pct:.1f}%", "#5BA876")
        kpi_card(c2, "😐", "Neutral %", f"{neu_pct:.1f}%", "#D9A441")
        kpi_card(c3, "☹️", "Negative %", f"{neg_pct:.1f}%", "#D96B6B")
        st.write("")
        c4, c5 = st.columns(2)
        kpi_card(c4, "⭐", "Average Rating", f"{avg_rating:.2f} / 5", PASTEL_PURPLE_DARK)
        kpi_card(c5, "💗", "Customer Satisfaction Index", f"{csi:.1f}%", PASTEL_PINK_DARK)

    with st.container(border=True):
        section_header("🏷️ Highlight Produk")
        c6, c7, c8 = st.columns(3)
        kpi_card(c6, "🌟", "Most Positive Product", f"{most_positive_product}", "#5BA876")
        kpi_card(c7, "⚠️", "Most Negative Product", f"{most_negative_product}", "#D96B6B")
        kpi_card(c8, "👍", "Most Helpful Product", f"{most_helpful_product}", PASTEL_BLUE_DARK)

    # CSI Gauge
    with st.container(border=True):
        section_header("📐 Customer Satisfaction Index (CSI)")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=csi,
            number={"suffix": "%", "font": {"color": TEXT_DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                "bar": {"color": PASTEL_PURPLE_DARK},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 50], "color": COLOR_NEGATIVE},
                    {"range": [50, 75], "color": COLOR_NEUTRAL},
                    {"range": [75, 100], "color": COLOR_POSITIVE},
                ],
            },
        ))
        st.plotly_chart(style_plotly(fig, 320), use_container_width=True)

    # Auto recommendation
    with st.container(border=True):
        section_header("🤖 Rekomendasi CRM Otomatis")

        if neg_pct > 30:
            st.markdown(f"""
            <div style="background:{COLOR_NEGATIVE}33; border-left:6px solid {COLOR_NEGATIVE}; padding:16px 20px; border-radius:14px; margin-bottom:10px;">
            <b>⚠️ Customer dissatisfaction is increasing.</b><br>
            Immediate service improvement is recommended. ({neg_pct:.1f}% reviews negatif)
            </div>
            """, unsafe_allow_html=True)

        if pos_pct > 70:
            st.markdown(f"""
            <div style="background:{COLOR_POSITIVE}33; border-left:6px solid {COLOR_POSITIVE}; padding:16px 20px; border-radius:14px; margin-bottom:10px;">
            <b>✅ Customer satisfaction is excellent.</b><br>
            Focus on retention and loyalty programs. ({pos_pct:.1f}% reviews positif)
            </div>
            """, unsafe_allow_html=True)

        if 30 >= neg_pct and pos_pct <= 70:
            st.markdown(f"""
            <div style="background:{COLOR_NEUTRAL}33; border-left:6px solid {COLOR_NEUTRAL}; padding:16px 20px; border-radius:14px; margin-bottom:10px;">
            <b>📊 Customer sentiment is moderate.</b><br>
            Pertahankan kualitas layanan dan pantau tren sentimen secara berkala untuk mencegah penurunan kepuasan.
            </div>
            """, unsafe_allow_html=True)



# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("✨ Amazon Customer Review Analytics — CRM & Sentiment Analysis Final Project | Built with Streamlit & Plotly")