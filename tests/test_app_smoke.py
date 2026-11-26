"""Runs app.py end to end against a tiny synthetic project, using a fake Streamlit."""

import runpy
import sys
from pathlib import Path

import pytest

from fake_streamlit import FakeStreamlit, StopApp

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def run_app(small_paths, monkeypatch):
    import engine as engine_module
    monkeypatch.setattr(engine_module, "DATA_PATH", small_paths["data_path"])
    monkeypatch.setattr(engine_module, "EXTRAS_PATH", small_paths["extras_path"])
    monkeypatch.setattr(engine_module, "TFIDF_PATH", small_paths["tfidf_path"])
    monkeypatch.setattr(engine_module, "SEMANTIC_PATH", small_paths["semantic_path"])

    def run(**script):
        fake = FakeStreamlit(**script)
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        sys.modules.pop("ui", None)           # make ui.py bind to the fake
        runpy.run_path(str(APP), run_name="__main__")
        return fake

    return run


def card_titles(fake):
    return [a for a in fake.rendered("markdown")
            if isinstance(a, str) and "<h4" in a]


def test_nothing_selected_shows_no_results(run_app):
    fake = run_app(selection_index=None, button=False)
    assert "result" not in fake.session_state
    assert fake.rendered("expander") == []


def test_recommendation_flow_renders_cards_and_reasons(run_app):
    # choices are sorted by title: Alpha is first
    fake = run_app(selection_index=0, how_many=5, genres=["Comedy"])
    result = fake.session_state["result"]
    assert result["seed"]["title"] == "Alpha"
    assert 1 <= len(result["recommendations"]) <= 5
    assert len(fake.rendered("expander")) == len(result["recommendations"])
    assert result["related"]
    assert any("Related Films" in str(a) for a in fake.rendered("markdown"))


def test_filters_reach_the_engine(run_app):
    fake = run_app(selection_index=0, industry="Bollywood")
    recs = fake.session_state["result"]["recommendations"]
    assert recs and all(r["industry"] == "Bollywood" for r in recs)


def test_no_matches_shows_warning(run_app):
    fake = run_app(selection_index=0, genres=["Nonexistent"])
    assert fake.rendered("warning")


def test_broken_models_show_error_not_traceback(run_app, monkeypatch, tmp_path):
    import engine as engine_module
    monkeypatch.setattr(engine_module, "TFIDF_PATH", tmp_path / "missing.pkl")
    with pytest.raises(StopApp):
        run_app()


def test_cards_escape_html_and_handle_missing_values(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    sys.modules.pop("ui", None)
    import ui
    ui.movie_card("<b>X & Y</b>", "", None, "", None)
    html = fake.rendered("markdown")[-1]
    assert "&lt;b&gt;X &amp; Y&lt;/b&gt;" in html
    assert "N/A" in html and "—" in html
