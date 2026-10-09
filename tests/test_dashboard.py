"""End-to-end tests of the Streamlit app with streamlit.testing (no browser needed)."""
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ["Overview", "Volatility forecast", "GMSI & volatility", "Market fragility (MFI)",
         "Shock propagation", "Methodology & limitations"]


def _run(page=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_fast_without_errors(page):
    t0 = time.time()
    at = _run(page)
    assert time.time() - t0 < 20, "page render too slow"
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _forecast_page():
    at = _run("Volatility forecast")
    return at


def test_historical_forecast_shows_result():
    at = _forecast_page()
    html = " ".join(m.value for m in at.markdown)
    assert "Forecast daily σ" in html and "Annualised" in html


def test_pasted_prices_validation_errors_are_friendly():
    at = _forecast_page()
    at.main.radio[0].set_value("Paste your own closing prices").run()
    at.text_area[0].input("100 101 abc").run()
    at.button[0].click().run()
    assert at.error and "not a number" in at.error[0].value
    at.text_area[0].input("100 101 102").run()
    at.button[0].click().run()
    assert at.error and "at least 61" in at.error[0].value


def test_pasted_prices_produce_forecast():
    at = _forecast_page()
    at.main.radio[0].set_value("Paste your own closing prices").run()
    at.text_area[0].input("\n".join(str(100 + (i % 7)) for i in range(80))).run()
    at.button[0].click().run()
    assert not at.error and not at.exception
    assert "Forecast daily σ" in " ".join(m.value for m in at.markdown)


def test_switching_asset_works():
    at = _forecast_page()
    at.selectbox[0].set_value("NIFTY").run()
    assert not at.exception and not at.error
