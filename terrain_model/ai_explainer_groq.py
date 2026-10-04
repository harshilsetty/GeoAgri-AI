"""Compatibility shim forwarding to src.explainability.ai_explainer."""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.explainability.ai_explainer import *
