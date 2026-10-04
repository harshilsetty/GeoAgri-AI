# GeoAgri-AI Computer Vision Subsystem (Planned)

## Target Architecture (KPIT Sparkle 2027)

The vision subsystem is designed for remote sensing ingestion:
- **Satellite Ingestion**: Sentinel-2 (10m-20m multispectral) and Landsat 8/9 bands.
- **Drone / Aerial Imagery**: High-resolution RGB and multispectral UAV orthomosaics.
- **Vegetation Indices**: Automated computation of NDVI, EVI, NDRE, and NDWI for canopy vigor and water content.
- **Canopy Segmentation & Stress Detection**: DeepLabV3+ and lightweight CNN/Transformer architectures adapted for agricultural field boundary detection, crop stress zoning, and canopy coverage estimation.

*Status: In design / roadmap for subsequent implementation phases. Legacy archaeological vision weights have been deprecated.*
