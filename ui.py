import html

import streamlit as st


def page_header():
    st.markdown("""
    <h1 style='text-align:center; font-size:55px;'>
        🎬 <span style='color:#c7161f;'>4</span><span style='color:#E50914;'>Cinephiles</span>
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h4 style='text-align:center; color:gray;'>
        Find your next favorite movie 🍿
    </h4>
    """, unsafe_allow_html=True)
    st.write("")


def hero_banner():
    st.markdown("""
    <div style="background: linear-gradient(90deg,#111,#1c1c1c,#111); padding:30px; border-radius:12px; text-align:center; margin-bottom:20px;">
        <h2 style="color:white;">
        Search Movies. Discover Your Next Favourite.</h2>
        <p style="color:#bbbbbb; font-size:18px;">Smarter movie discovery with AI-powered recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.image("assets/moviesbanner.png", use_container_width=True)
    st.write("")


def _clean(value, fallback="—"):
    """Escape text for HTML; show a dash for missing values."""
    if value is None:
        return fallback
    text = str(value).strip()
    return html.escape(text) if text and text.lower() != "nan" else fallback


def format_rating(rating):
    try:
        return f"{float(rating):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def format_runtime(runtime):
    try:
        return f"{int(runtime)} min"
    except (TypeError, ValueError):
        return "—"


def movie_card(title, genres, rating, year, runtime):
    """Original card. Values are HTML-escaped and missing values are handled."""
    st.markdown(f"""
    <div style="
        background:#1c1c1c; border-radius:15px; padding:12px; text-align:center; min-height:110px; box-shadow:0px 3px 8px rgba(0,0,0,0.3);
    ">
        <h4 style="color:white; margin-bottom:10px;">{_clean(title)}</h4>
        <p style="color:#FFD700; font-size:16px;">⭐ {format_rating(rating)}</p>
        <p style="color:#BBBBBB;">📅 {_clean(year)}</p>
        <p style="color:#BBBBBB;">⏱ {format_runtime(runtime)}</p>
        <span style="color:#E50914; font-size:15px;">🎭 {_clean(genres)}</span>
    </div>
    """, unsafe_allow_html=True)


def industry_badge(industry):
    colour = "#E50914" if industry == "Bollywood" else "#3b82f6"
    st.markdown(
        f"<div style='text-align:center; margin:6px 0 4px 0;'>"
        f"<span style='background:{colour}; color:white; padding:2px 10px; "
        f"border-radius:10px; font-size:12px;'>{_clean(industry)}</span></div>",
        unsafe_allow_html=True,
    )


def section_title(text, colour="#E50914"):
    st.markdown(
        f"<h2 style='color:{colour};'>{_clean(text)}</h2>",
        unsafe_allow_html=True,
    )


def detail_block(movie):
    """Text part of the selected-movie panel."""
    genres = " • ".join(movie["genres"]) if movie["genres"] else "—"
    st.markdown(f"""
    <div style="background:#1c1c1c; border-radius:15px; padding:18px;">
        <h2 style="color:white; margin-bottom:4px;">{_clean(movie['title'])}</h2>
        <p style="color:#BBBBBB;">{_clean(movie['industry'])} · {_clean(movie['year'])}
           · ⏱ {format_runtime(movie['runtime'])} · ⭐ {format_rating(movie['rating'])}</p>
        <p style="color:#E50914;">🎭 {_clean(genres)}</p>
        <p style="color:#DDDDDD;">{_clean(movie['overview'], 'No overview available.')}</p>
        <p style="color:#BBBBBB;">🎬 Director: {_clean(movie['director'])}</p>
        <p style="color:#BBBBBB;">👥 Cast: {_clean(movie['cast'])}</p>
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:gray;">
        Made with ❤️ by Abdus
    </div>
    """, unsafe_allow_html=True)
