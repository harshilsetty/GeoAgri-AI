"""Root compatibility shim for agri_features. Points to src.preprocessing.agri_features."""
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing.agri_features import *
