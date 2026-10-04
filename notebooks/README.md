# ?? GeoAgri-AI Research & Exploration Notebooks

This directory is designated for interactive Jupyter notebooks used during exploratory data analysis, model prototyping, and evaluation visualization.

---

## Structure & Guidelines

1. **Clean Version Control**: All committed notebooks must have cell outputs cleared before committing, or use automated git hooks to strip output caches.
2. **Reproducibility**: Notebooks should set fixed random seeds (`SEED = 42`) and load data using repository-relative paths (`Path(__file__).parents[...]` or relative to repository root).
3. **Planned Notebooks**:
   * `01_geospatial_feature_analysis.ipynb`: Visualizing soil and climate distributions across Indian agro-climatic zones.
   * `02_crop_suitability_benchmarking.ipynb`: Benchmarking multi-class XGBoost models against LightGBM and Random Forest baselines.
   * `03_moisture_stress_modeling.ipynb`: Prototyping dynamic water deficit and evapotranspiration stress indicators.
   * `04_explainability_deep_dive.ipynb`: TreeSHAP summary plots, interaction values, and counterfactual decision boundaries.
