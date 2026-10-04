"""Root compatibility shim for backend_api. Points to src.api.backend_api."""
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.api.backend_api import *
