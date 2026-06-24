import ast
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "movie_data.csv"
TOP_N = 5

st.set_page_config(
    page_title="RecoMovie · Content-Based Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f64f59, #c471ed, #12c2e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: #a0a0c0;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    .movie-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 12px;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(198, 113, 237, 0.35);
        border-color: rgba(198, 113, 237, 0.5);
    }
    .movie-card img { border-radius: 10px; width: 100%; object-fit: cover; }
    .movie-title {
        margin-top: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e8e8ff;
        line-height: 1.3;
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(90deg, #f64f59, #c471ed);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-bottom: 6px;
    }

    .selected-panel {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 1.5rem;
    }
    .selected-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #c471ed;
        margin-bottom: 4px;
    }
    .selected-title { font-size: 1.6rem; font-weight: 700; color: #ffffff; line-height: 1.2; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c471ed;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(196, 113, 237, 0.3);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    label { color: #c0c0e0 !important; font-size: 0.9rem !important; }

    .no-poster {
        height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        color: #666;
        font-size: 0.8rem;
    }

    .info-pill {
        display: inline-block;
        background: rgba(18, 194, 233, 0.15);
        border: 1px solid rgba(18, 194, 233, 0.3);
        color: #12c2e9;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_list_column(value: str) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip().lower().replace(" ", "") for v in value]

    if not isinstance(value, str) or value.strip() in ("", "[]", "{}"):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v).strip().lower().replace(" ", "") for v in parsed]
        if isinstance(parsed, dict):
            return [str(v).strip().lower().replace(" ", "") for v in parsed.values()]
    except (ValueError, SyntaxError):
        pass

    cleaned = re.sub(r"[\[\]'\"{}]", "", value)
    return [w.strip().lower().replace(" ", "") for w in cleaned.split(",") if w.strip()]


@st.cache_data(show_spinner="🎬 Loading movie database and building feature matrix…")
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["title", "genres", "keywords", "director", "poster_url"]].copy()
    df.dropna(subset=["title", "genres", "keywords", "director"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["genres_parsed"] = df["genres"].apply(parse_list_column)
    df["keywords_parsed"] = df["keywords"].apply(parse_list_column)
    df["director_tag"] = df["director"].fillna("").apply(lambda name: name.strip().replace(" ", ""))

    def build_tags(row) -> str:
        tokens = row["genres_parsed"] + row["keywords_parsed"] + [row["director_tag"]] * 3
        return " ".join(tokens).lower()

    df["combined_tags"] = df.apply(build_tags, axis=1)
    df = df[df["combined_tags"].str.strip() != ""].reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_vectorizer(tags_series: tuple) -> tuple:
    vectorizer = CountVectorizer(stop_words="english", max_features=10_000)
    count_matrix = vectorizer.fit_transform(tags_series)
    return vectorizer, count_matrix


def get_recommendations(df: pd.DataFrame, count_matrix, movie_title: str, top_n: int = TOP_N) -> pd.DataFrame:
    matches = df[df["title"] == movie_title]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    target_vector = count_matrix[idx]
    sim_scores = cosine_similarity(target_vector, count_matrix).flatten()
    ranked_indices = sim_scores.argsort()[::-1]
    ranked_indices = [i for i in ranked_indices if i != idx][:top_n]
    return df.iloc[ranked_indices].reset_index(drop=True)


def render_poster(url: str | None, title: str) -> None:
    if url and isinstance(url, str) and url.startswith("http"):
        st.image(url, use_container_width=True, caption=None)
    else:
        st.markdown('<div class="no-poster">🎞️ No Poster Available</div>', unsafe_allow_html=True)


def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 20px;">
                <div style="font-size:3rem;">🎬</div>
                <div style="font-size:1.2rem; font-weight:700; color:#c471ed;">RecoMovie</div>
                <div style="font-size:0.75rem; color:#888; margin-top:4px;">Content-Based Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-header">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.82rem; color:#a0a0c0; line-height:1.7;">
            This engine uses <b style="color:#c471ed;">Content-Based Filtering</b>.<br><br>
            Each movie is described by a <em>bag-of-words</em> built from its genres, keywords, and director.<br><br>
            We use <b>CountVectorizer</b> + <b>Cosine Similarity</b> to find movies whose descriptions are
            closest to the one you pick — <em>without</em> computing a huge N×N matrix.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown('<div class="section-header">Selected Movie</div>', unsafe_allow_html=True)
        sidebar_poster_slot = st.empty()
        sidebar_title_slot = st.empty()

    st.markdown(
        '<p class="hero-title">🎬 RecoMovie</p>'
        '<p class="hero-sub">Discover your next favourite film with Content-Based Filtering · '
        'powered by scikit-learn & Streamlit</p>',
        unsafe_allow_html=True,
    )

    try:
        df = load_and_preprocess(DATASET_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at `{DATASET_PATH}`. Please place your CSV in the same directory as `app.py`.")
        st.stop()

    _, count_matrix = build_vectorizer(tuple(df["combined_tags"].tolist()))

    movie_titles = sorted(df["title"].unique().tolist())
    default_index = movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0

    col_sel, col_info = st.columns([3, 2])
    with col_sel:
        selected_title = st.selectbox(
            "🔍  Search for a movie",
            options=movie_titles,
            index=default_index,
            key="movie_selector",
            help="Type to filter the list of movies",
        )
    with col_info:
        st.markdown(
            f'<div style="padding-top:1.8rem; color:#a0a0c0; font-size:0.82rem;">'
            f'📦 <b style="color:#e0e0ff;">{len(df):,}</b> movies indexed &nbsp;·&nbsp; '
            f'Top <b style="color:#e0e0ff;">{TOP_N}</b> recommendations</div>',
            unsafe_allow_html=True,
        )

    if not selected_title:
        st.info("👆 Pick a movie above to get personalised recommendations.")
        st.stop()

    selected_movie = df[df["title"] == selected_title].iloc[0]

    with sidebar_poster_slot:
        render_poster(selected_movie.get("poster_url"), selected_title)
    with sidebar_title_slot:
        st.markdown(
            f'<div style="text-align:center; font-size:0.85rem; font-weight:600; color:#e8e8ff; margin-top:8px;">'
            f'{selected_title}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="selected-panel">', unsafe_allow_html=True)
    banner_img, banner_text = st.columns([1, 4])

    with banner_img:
        render_poster(selected_movie.get("poster_url"), selected_title)

    with banner_text:
        st.markdown(
            '<div class="selected-label">Now viewing</div>'
            f'<div class="selected-title">{selected_title}</div>',
            unsafe_allow_html=True,
        )
        genres_parsed = parse_list_column(str(selected_movie.get("genres", "")))
        director = str(selected_movie.get("director", "Unknown"))
        genre_pills = "".join(f'<span class="info-pill">{g.title()}</span>' for g in genres_parsed[:6])
        st.markdown(
            f'<div style="margin-top:12px;">'
            f'  <span style="color:#888; font-size:0.75rem;">🎭 Genres &nbsp;</span>{genre_pills}'
            f'</div>'
            f'<div style="margin-top:10px; color:#a0a0c0; font-size:0.82rem;">🎬 <b>Director:</b> {director}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("🤖 Finding similar movies…"):
        recommendations = get_recommendations(df, count_matrix, selected_title, TOP_N)

    if recommendations.empty:
        st.warning("⚠️ Could not generate recommendations for this movie.")
        st.stop()

    st.markdown(
        f'<div class="section-header" style="margin-top:1.5rem;">'
        f'🍿 Top {TOP_N} Recommendations based on <em>{selected_title}</em></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(TOP_N)
    for rank, (col, (_, row)) in enumerate(zip(cols, recommendations.iterrows()), start=1):
        with col:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(f'<div><span class="rank-badge">#{rank}</span></div>', unsafe_allow_html=True)
            render_poster(row.get("poster_url", None), row["title"])
            st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.72rem; color:#888; margin-top:4px;">🎬 {row.get("director", "")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; margin-top:3rem; padding:1.5rem;
                    border-top: 1px solid rgba(255,255,255,0.08); color:#555; font-size:0.75rem;">
            RecoMovie · Content-Based Filtering · Built with
            <span style="color:#c471ed;">Streamlit</span> &amp;
            <span style="color:#12c2e9;">scikit-learn</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
