"""
llm_client.py - one place to choose the LLM backend.

Switch backends with environment variables, no code changes:

  Groq (default, current behaviour):
      GROQ_API_KEY=gsk_...

  Self-hosted vLLM or Ollama (anything OpenAI-compatible):
      LLM_BASE_URL=https://<pod-proxy-url>/v1
      LLM_API_KEY=<the key you set for vLLM; any text for Ollama>
      LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ        (vLLM)
      LLM_MODEL=qwen2.5:7b-instruct-q4_K_M          (Ollama)

Requires: pip install openai
"""

import os

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_client_and_model(default_model: str = "openai/gpt-oss-20b"):
    """Return (client, model_name, backend) where backend is 'groq' or 'self-hosted'."""
    base_url = os.environ.get("LLM_BASE_URL")

    if base_url:
        backend = "self-hosted"
        api_key = os.environ.get("LLM_API_KEY", "none")
    else:
        backend = "groq"
        base_url = GROQ_BASE_URL
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No LLM configured. Set GROQ_API_KEY (Groq), or LLM_BASE_URL + "
                "LLM_API_KEY + LLM_MODEL (self-hosted vLLM/Ollama)."
            )

    model = os.environ.get("LLM_MODEL", default_model)

    # max_retries=0: the pipeline already has its own retry loop.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180, max_retries=0)
    return client, model, backend


def completion_kwargs(model: str, backend: str) -> dict:
    """Backend-specific generation settings.

    gpt-oss on Groq is a reasoning model: hidden reasoning tokens share the output
    budget, so it needs a big budget and low reasoning effort. Plain chat models
    (Qwen etc.) need neither, and reject or ignore reasoning_effort.
    """
    if backend == "groq" and model.startswith("openai/gpt-oss"):
        return {
            "max_completion_tokens": 4096,
            "extra_body": {"reasoning_effort": "low"},
        }
    return {"max_tokens": 1024}
