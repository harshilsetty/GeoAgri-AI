"""Root compatibility shim for geo_features. Points to src.geospatial.geo_features."""
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.geospatial.geo_features import *
