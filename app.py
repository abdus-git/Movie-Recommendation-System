import streamlit as st

from engine import RecommendationEngine
from ui import (
    page_header, hero_banner, movie_card, footer,
    industry_badge, section_title, detail_block,
)

st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

## css
st.markdown("""
<style>

.stButton>button{
    background:#E50914;
    color:white;
    border-radius:12px;
    height:50px;
    width:180px;
    font-weight:bold;
    border:none;
    transition:0.3s;
}
.stButton > button:hover{
    background:#B20710;
}
div[data-baseweb="select"]{
    border-radius:10px;
}
[data-testid="stImage"] img{
    border-radius:15px;
    object-fit:cover;
}
</style>
""", unsafe_allow_html=True)

NO_POSTER = "assets/no_poster.png"
CARDS_PER_ROW = 5
ALL_INDUSTRIES = "All"


@st.cache_resource(show_spinner="Loading movie data and models...")
def load_engine():
    return RecommendationEngine()


try:
    engine = load_engine()
except (FileNotFoundError, OSError, EOFError, ValueError) as error:
    st.error(f"Could not load the recommendation engine: {error}")
    st.stop()

page_header()
hero_banner()

##sidebar
stats = engine.stats()

st.sidebar.title("🎬 Movie Recommendation")
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write(
    """
    4Cinephiles is an AI-powered Movie Recommendation System
    for Bollywood and Hollywood films.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Recommendation Method")
st.sidebar.success("Hybrid: TF-IDF + semantic embeddings, genre-aware")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")
st.sidebar.metric("Movies", stats["total"])
left, right = st.sidebar.columns(2)
left.metric("Bollywood", stats.get("Bollywood", 0))
right.metric("Hollywood", stats.get("Hollywood", 0))
st.sidebar.caption(
    "'Hollywood' means the TMDB 5000 set, which includes some "
    "non-English films."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
industry_choice = st.sidebar.radio(
    "Recommend films from",
    [ALL_INDUSTRIES, "Bollywood", "Hollywood"],
)
genre_choice = st.sidebar.multiselect(
    "Genres (any of)",
    engine.genres,
)
how_many = st.sidebar.select_slider(
    "Number of recommendations",
    options=[5, 10, 15, 20],
    value=10,
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Abdus ❤️")


st.markdown("""
<h3 style="color:white;">
🎥 Choose a Movie
</h3>
""", unsafe_allow_html=True)

choices = engine.choices()
label_to_id = dict(choices)

selected_label = st.selectbox(
    "Search by title",
    [label for label, _ in choices],
    index=None,
    placeholder="Type a movie name...",
    label_visibility="collapsed",
)

selected_id = label_to_id.get(selected_label)
industry_filter = None if industry_choice == ALL_INDUSTRIES else industry_choice

if selected_id is not None:
    movie = engine.details(selected_id)

    poster_col, info_col = st.columns([1, 3])
    with poster_col:
        st.image(movie["poster_url"] or NO_POSTER, use_container_width=True)
    with info_col:
        detail_block(movie)

col1, col2, col3 = st.columns([2, 2, 1])

with col2:
    recommend_button = st.button("🎬 Recommend", disabled=selected_id is None)

request = (selected_id, industry_filter, tuple(genre_choice), how_many)

if recommend_button and selected_id is not None:

    with st.spinner("Finding similar movies... 🍿"):
        st.session_state["result"] = engine.recommend(
            selected_id,
            top_n=how_many,
            industry=industry_filter,
            genres=list(genre_choice),
        )
        st.session_state["request"] = request

result = st.session_state.get("result")

if result is not None:

    if st.session_state.get("request") != request:
        st.info("Your selection or filters changed. Press Recommend to update.")

    seed = result["seed"]
    recommendations = result["recommendations"]

    section_title(f"🍿 Because you liked {seed['title']}")

    if not recommendations:
        st.warning(
            "No recommendations match these filters. "
            "Try 'All' industries or fewer genres."
        )

    for start in range(0, len(recommendations), CARDS_PER_ROW):
        row = recommendations[start:start + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)

        for col, rec in zip(cols, row):
            with col:
                st.image(rec["poster_url"] or NO_POSTER,
                         use_container_width=True)
                industry_badge(rec["industry"])
                movie_card(
                    rec["title"],
                    " • ".join(rec["genres"][:3]),
                    rec["rating"],
                    rec["year"],
                    rec["runtime"],
                )
                with st.expander("Why this movie?"):
                    for reason in rec["reasons"]:
                        st.write("• " + reason)
                    st.caption(
                        f"Ranking score {rec['score']} "
                        "(a ranking signal, not a probability)."
                    )

    related = result["related"]
    if related:
        collection = related[0]["collection_name"] or "the same franchise"
        section_title(f"🎞️ Related Films · {collection}", colour="#BBBBBB")

        for start in range(0, len(related), CARDS_PER_ROW):
            row = related[start:start + CARDS_PER_ROW]
            cols = st.columns(CARDS_PER_ROW)

            for col, rec in zip(cols, row):
                with col:
                    st.image(rec["poster_url"] or NO_POSTER,
                             use_container_width=True)
                    movie_card(
                        rec["title"],
                        " • ".join(rec["genres"][:3]),
                        rec["rating"],
                        rec["year"],
                        rec["runtime"],
                    )

footer()
