# GEO AI Crop Intelligence V5 -- Error Diagnostic Analysis

## Summary of Recommendation Errors
- **Total Evaluated Chronological Test Instances:** 546
- **Incorrect Top-1 Recommendations:** 443 (81.14%)
- **True Crop in Top-3 (Near-Boundary Tie):** 128 (28.9%)
- **True Crop in Top-5 (Agronomically Feasible):** 215 (48.5%)

## Categorization
1. **Agro-climatic Niche Overlap (True crop in Top-3):** 27.2% of errors. Conditions strongly support multiple crops with overlapping thermal and moisture requirements (e.g., Maize vs Rice, Moong vs Urad).
2. **Broad Ecological Compatibility (True crop in Top-5):** 24.1% of errors. Multiple viable alternatives exist under Indian monsoon conditions.
3. **Severe Pedological Divergence:** 48.7% of errors. Specific soil micronutrient or alkalinity preferences.
