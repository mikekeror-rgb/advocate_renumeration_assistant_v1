"""Shared test setup: make the project importable, and load eval/eval.py without the
heavy RAG pipeline (Chroma, sentence-transformers, torch), which unit tests don't need."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def eval_module():
    """eval/eval.py imports the pipeline at import time. Its checking functions are pure,
    so a tiny stand-in pipeline module is used in its place."""
    stub = types.ModuleType("stub_pipeline")
    stub.RagPipeline = object
    stub.CONTEXT_SNIPPET_CHARS = 1000
    sys.modules["stub_pipeline"] = stub
    os.environ["PIPELINE_MODULE"] = "stub_pipeline"
    return load_module("eval_harness", ROOT / "eval" / "eval_v2.py")


@pytest.fixture(scope="session")
def bench_module():
    return load_module("bench", ROOT / "scripts" / "bench.py")
