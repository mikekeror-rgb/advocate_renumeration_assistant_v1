"""Visit a Streamlit Community Cloud app and wake it if it is asleep.

Reads the app URL from the STREAMLIT_APP_URL environment variable.
Exits non-zero only if the app never becomes reachable, so the workflow
turns red when the app is genuinely down.
"""
import os
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
WAKE_BUTTON = "button:has-text('Yes, get this app back up')"


def main() -> int:
    if not URL:
        print("STREAMLIT_APP_URL is not set (repo Settings > Variables).")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)

        # Give the page a few seconds to render either the app or the sleep screen.
        page.wait_for_timeout(8_000)

        button = page.locator(WAKE_BUTTON)
        if button.count() > 0:
            print("App was asleep: clicking the wake button.")
            button.first.click()
            try:
                # The wake screen disappears once the app starts booting.
                button.first.wait_for(state="detached", timeout=60_000)
            except PWTimeout:
                print("Wake button still visible after 60s.")
                browser.close()
                return 1
            page.wait_for_timeout(60_000)  # let the app finish starting
        else:
            print("App was already awake.")

        # Streamlit renders its app container once the app is running.
        try:
            page.wait_for_selector("[data-testid='stApp']", timeout=60_000)
            print("App is up.")
            ok = True
        except PWTimeout:
            print("App container never appeared.")
            ok = False

        browser.close()
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
