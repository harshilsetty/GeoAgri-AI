"""Root compatibility shim for agri_inference. Points to src.models.agri_inference."""
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.agri_inference import *
