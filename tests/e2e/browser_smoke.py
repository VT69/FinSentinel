"""
Browser smoke test against a RUNNING dashboard (not collected by pytest).

    streamlit run dashboard/app.py --server.headless true --server.port 8501 &
    pip install playwright && python tests/e2e/browser_smoke.py [URL]
Env: CHROMIUM=/path/to/chrome to use a preinstalled browser instead of `playwright install chromium`.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
PAGES = ["Overview", "Volatility forecast", "GMSI & volatility", "Market fragility (MFI)",
         "Shock propagation", "Methodology & limitations"]


def main() -> int:
    failures = 0
    with sync_playwright() as p:
        kw = {"executable_path": os.environ["CHROMIUM"]} if os.environ.get("CHROMIUM") else {}
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        console_errors = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        t0 = time.time()
        page.goto(URL, wait_until="networkidle")
        page.get_by_text("What the data actually shows").wait_for(timeout=60_000)
        print(f"first paint: {time.time() - t0:.1f}s")
        sidebar = page.locator("section[data-testid='stSidebar']")
        for name in PAGES:
            t0 = time.time()
            sidebar.get_by_text(name, exact=True).click()
            page.wait_for_timeout(1500)
            exc = page.locator("[data-testid='stException']").count()
            err = page.locator("[data-testid='stAlertContentError']").count()
            charts = page.locator(".js-plotly-plot").count()
            failures += exc + err
            print(f"{name:28s} {time.time() - t0:4.1f}s  h1={page.locator('h1').first.inner_text()!r}  "
                  f"charts={charts} exceptions={exc} errors={err}")
            if name == "Volatility forecast":
                print("  historical:", page.locator(".kcard").first.inner_text().replace("\n", " | "))
                page.get_by_text("Paste your own closing prices").click()
                page.wait_for_timeout(800)
                box = page.locator("textarea").first
                box.fill("100 101 abc")
                page.get_by_role("button", name="Forecast").click()
                page.wait_for_timeout(1500)
                msg = page.locator("[data-testid='stAlertContentError']").first.inner_text()
                print("  bad input ->", msg)
                failures += "not a number" not in msg
                box.fill("\n".join(str(100 + (i % 7) * 0.5) for i in range(80)))
                page.get_by_role("button", name="Forecast").click()
                page.wait_for_timeout(1500)
                print("  good input ->", page.locator(".kcard").first.inner_text().replace("\n", " | "))
                failures += page.locator("[data-testid='stAlertContentError']").count()
        print(f"browser console errors: {len(console_errors)}")
        failures += len(console_errors)
        browser.close()
    print("PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
