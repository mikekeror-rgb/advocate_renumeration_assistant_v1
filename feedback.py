"""
Document-request suggestions, filed as GitHub issues.

Streamlit Community Cloud's filesystem resets on every restart/redeploy, so
suggestions can't be saved to a local file. Each suggestion becomes an issue
labelled "document-request" in the repo instead, which also works as a to-do
list: close the issue once the document has been added to the corpus.

Setup (Streamlit Cloud → app → Settings → Secrets, or .streamlit/secrets.toml locally):
    GITHUB_TOKEN = "github_pat_..."   # fine-grained token, this repo only, Issues: Read and write
    GITHUB_REPO  = "owner/repo-name"
"""

import os

import requests

LABEL = "document-request"


def _setting(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets file (e.g. running outside Streamlit) — fall back to env vars
    return os.environ.get(name)


def is_configured() -> bool:
    return bool(_setting("GITHUB_TOKEN") and _setting("GITHUB_REPO"))


def _as_code_block(text: str) -> str:
    """Put user text in a code block so it can't @mention people, inject
    markdown, or render links/images inside the issue."""
    return "```\n" + text.replace("```", "'''") + "\n```"


def submit_document_request(question: str, suggestion: str, link: str = "") -> str:
    """Create a GitHub issue for the suggestion. Returns the issue URL.
    Raises RuntimeError with a readable message if anything goes wrong."""
    token, repo = _setting("GITHUB_TOKEN"), _setting("GITHUB_REPO")
    if not (token and repo):
        raise RuntimeError("Suggestions aren't configured (GITHUB_TOKEN / GITHUB_REPO missing).")

    short_q = " ".join(question.split())
    title = f"Document request: {short_q[:80]}{'…' if len(short_q) > 80 else ''}"
    body = "\n\n".join([
        "**Question the assistant couldn't answer:**",
        _as_code_block(question),
        "**Suggested document or ruling:**",
        _as_code_block(suggestion),
        "**Link (unverified, user-supplied):**",
        _as_code_block(link or "none given"),
        "_Submitted from the app's suggestions box._",
    ])

    try:
        resp = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"title": title, "body": body, "labels": [LABEL]},
            timeout=15,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Couldn't reach GitHub: {e}") from e

    if resp.status_code != 201:
        raise RuntimeError(f"GitHub returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()["html_url"]
