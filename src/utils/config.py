"""Configuration module for GeoAgri-AI."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "agriculture"
DEFAULT_CROP_MODEL_VERSION = os.getenv("GEO_AI_CROP_MODEL_VERSION", "v2").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
