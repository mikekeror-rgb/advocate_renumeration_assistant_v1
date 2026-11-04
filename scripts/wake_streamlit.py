"""Visit a Streamlit Community Cloud app and wake it if it is asleep.

Reads the app URL from the STREAMLIT_APP_URL environment variable.
Exits non-zero only if the app never becomes reachable, so the workflow
turns red when the app is genuinely down. On failure it prints what the
page actually showed and saves wake_failure.png for the workflow to upload.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
WAKE_BUTTON = "button:has-text('get this app back up')"
APP_SELECTOR = "[data-testid='stApp']"
ERROR_TEXTS = ("Error running app", "Oh no.")


def app_frame(page):
    """The app usually runs inside an iframe on *.streamlit.app, so search
    every frame instead of relying on the iframe's title."""
    for frame in page.frames:
        try:
            if frame.locator(APP_SELECTOR).count() > 0:
                return frame
        except Exception:
            pass  # frame was detached mid-check
    return None


def wait_for_app(page, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if app_frame(page):
            return True
        page.wait_for_timeout(3_000)
    return False


def describe(page):
    print(f"  final URL: {page.url}")
    print(f"  title: {page.title()!r}")
    for f in page.frames:
        print(f"  frame: {f.url[:120]}")
    for f in page.frames:
        try:
            text = f.locator("body").inner_text(timeout=2_000)
        except Exception:
            continue
        for err in ERROR_TEXTS:
            if err in text:
                print(f"  APP ERROR PAGE DETECTED: {text[:300]!r}")
                return
    try:
        body = page.locator("body").inner_text(timeout=2_000)
        print(f"  visible text: {body[:300]!r}")
    except Exception:
        pass


def main() -> int:
    if not URL:
        print("STREAMLIT_APP_URL is not set (repo Settings > Secrets and variables > Actions > Variables).")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for attempt in (1, 2):
            try:
                page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
                break
            except Exception as e:
                print(f"Load attempt {attempt} failed: {e}")
                if attempt == 2:
                    browser.close()
                    return 1
                page.wait_for_timeout(10_000)

        page.wait_for_timeout(8_000)

        button = page.locator(WAKE_BUTTON)
        if button.count() > 0:
            print("App was asleep: clicking the wake button.")
            button.first.click()
            timeout = 240  # cold starts (model + index loading) can take minutes
        else:
            print("No wake button: app should be awake.")
            timeout = 90

        ok = wait_for_app(page, timeout)
        if ok:
            print("App is up.")
        else:
            print(f"App container not found after {timeout}s. What the page showed:")
            describe(page)
            page.screenshot(path="wake_failure.png", full_page=True)

        browser.close()
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
