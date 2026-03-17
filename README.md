# Borehole Water Level Prediction — Sidama, Gofa & Gamo (Ethiopia)

This project predicts groundwater Static Water Level (SWL) in boreholes across the Sidama, Gofa, and Gamo zones of Ethiopia. It combines satellite remote sensing data (precipitation, LST, NDVI, humidity, wind speed) with terrain and soil data to train a Gradient Boosting machine learning model.

This project was part of the **World Map Project**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Features Used in the Model](#features-used-in-the-model)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [References](#references)

---

## Project Overview

The goal is to map and predict SWL at a dense grid of locations across the Sidama region using:

- Observed borehole SWL measurements
- Satellite-derived environmental variables averaged over three climatic seasons
- Terrain (DEM) and soil type data

Three seasons are used throughout the pipeline:

| Season | Months | Label |
|---|---|---|
| Big Rains | June – September | `JunToSep` |
| Dry Season | October – January | `OctToJan` |
| Small Rains | February – May | `FebToMay` |

---

## Data Sources

### DEM (Digital Elevation Model)
- **Source**: USGS EarthExplorer — SRTM 1 Arc-Second Global (30 m resolution)
- **Access**: [earthexplorer.usgs.gov](https://earthexplorer.usgs.gov)
- Navigate to: *Digital Elevation → SRTM → SRTM 1 Arc-Second Global*

### Soil Types
- **Source**: FAO Harmonized World Soil Database v1.2
- **Access**: [FAO Soils Portal](https://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/)
- Download the HWSD Raster (GeoTIFF) for soil type classification

### Precipitation
- **Product**: GPM_3IMERGM v07 — Global monthly precipitation (HDF5)
- **Resolution**: 0.1° (~10 km)
- **Period used**: Oct 2005 – Sep 2007 (two years)
- **Unit conversion**: mm/hr × 24 hr/day × days/month → cumulative mm/month
- **Access**: [NASA Earthdata](https://search.earthdata.nasa.gov/search)
- **Reference**: Huffman et al. (2019). *GPM IMERG Final Precipitation L3 1 month 0.1 degree V06*. GES DISC.

### Specific Humidity & Wind Speed
- **Product**: FLDAS_NOAH01_C_GL_M — FLDAS Noah Land Surface Model (NetCDF)
- **Resolution**: 0.1° (~10 km), monthly
- **Period used**: Oct 2005 – Sep 2007 (two years)
- **Variables**: `Qair_f_tavg` (specific humidity), `Wind_f_tavg` (wind speed)
- **Access**: [NASA Earthdata](https://search.earthdata.nasa.gov/search)
- **Reference**: Amy McNally NASA/GSFC/HSL (2018). *FLDAS Noah Land Surface Model L4 Global Monthly 0.1 x 0.1 degree*. GES DISC.

### Land Surface Temperature (LST)
- **Product**: MOD11A1 v006 — MODIS/Terra LST/Emissivity Daily (TIF/HDF)
- **Resolution**: 1 km, daily
- **Period used**: Oct 2005 – Sep 2007 (two years)
- **Scale factor**: × 0.02 to convert to °C
- **Variables**: LST Daytime and Nighttime
- **Access**: [NASA Earthdata](https://search.earthdata.nasa.gov/search)
- **Reference**: Wan, Z., Hook, S., Hulley, G. (2015). *MOD11A1 MODIS/Terra Land Surface Temperature/Emissivity Daily L3 Global 1km SIN Grid V006*. NASA EOSDIS LP DAAC.

### NDVI
- **Product**: MOD13Q1 v006 — MODIS/Terra Vegetation Indices 16-Day (TIF/HDF)
- **Resolution**: 250 m, every 16 days
- **Period used**: Oct 2006 – Sep 2007 (one year)
- **Scale factor**: × 0.0001 to obtain final NDVI values (range −1 to 1)
- **Access**: [NASA Earthdata](https://search.earthdata.nasa.gov/search)
- **Reference**: Didan, K. (2015). *MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V006*. NASA EOSDIS LP DAAC.

---

## Repository Structure

```
Final Project files/
├── Code (Python)/
│   ├── Data extraction scripts/
│   │   ├── 1-MODL_ExtractPrecipitationNASA.py          # GPM precipitation extraction
│   │   ├── 2-MODL_ExtractSpecificHumidityAndWindSpeedFromFLDAS.py  # FLDAS humidity & wind
│   │   ├── 3-MODL_ExtractLST_MODIS.py                  # MODIS LST extraction
│   │   ├── 4-MODL_ExtractNDVI250mFromNASA.py            # MODIS NDVI extraction
│   │   ├── gpm_precipitation_extractor.py               # Alternative precipitation extractor
│   │   ├── ExtractPrecipitation.py                      # Precipitation helper
│   │   ├── test_coordinates.py                          # Coordinate validation utility
│   │   ├── SidamaGridpointsExtractor/
│   │   │   └── gridpointsforpredictionextractor.py      # Generate prediction grid points
│   │   └── Use_final_datasetmaker_after_individual_extractions/
│   │       └── FinalDatasetMaker.py                     # Merge all datasets into one CSV
│   ├── DataAnalysisCode/
│   │   └── finaldatadiagnostics.py                      # Dataset diagnostics & QC
│   ├── Interpolators_scripts/
│   │   ├── PrecipitationSpatialInterpolation.py         # IDW interpolation for precipitation
│   │   └── Clusterinterpolator_NDVI_LST_Wind_Humidity.py  # Fast KDTree IDW interpolation
│   └── Model for water level prediction/
│       ├── Final_Model_For_Sidama_predictions.py        # Model training (v1)
│       └── Final_Model_For_Prediction_version_2.py      # Model training + prediction (v2)
├── DataDescription.docx                                 # Detailed data source documentation
├── Boreholes Sidama, Gofa, Gamo.xlsx                   # Original borehole SWL records
├── SidamaGridPoints_Predicted.csv                       # Final predicted SWL at grid points
├── requirements.txt                                     # Python dependencies
├── Shapefile/                                           # Study area boundary shapefiles
├── ExtractedData/                                       # Raw extracted variable CSVs
├── InterpolatedData/                                    # Spatially interpolated datasets
├── ModelData/                                           # Merged dataset for model training
├── TrainedModel/                                        # Saved model artifacts (.pkl)
└── TestDataForPrediction/                               # Grid point data for prediction
```

---

## Pipeline Workflow

The full pipeline runs in the following order:

### Step 1 — Generate Prediction Grid Points
```
SidamaGridpointsExtractor/gridpointsforpredictionextractor.py
```
Generates a regular grid of points (0.01° spacing, ~1 km) within the Sidama boundary shapefile. Outputs a CSV with `longitude` and `latitude` columns used as input for all extraction scripts.

### Step 2 — Extract Environmental Variables (run independently)

| Script | Input | Output |
|---|---|---|
| `1-MODL_ExtractPrecipitationNASA.py` | GPM HDF5 files + grid points CSV | `SeasonalPrecipitation.csv` |
| `2-MODL_ExtractSpecificHumidityAndWindSpeedFromFLDAS.py` | FLDAS NetCDF files + shapefile | Seasonal humidity & wind CSVs |
| `3-MODL_ExtractLST_MODIS.py` | MODIS MOD11A1 TIF files + grid points CSV | `LSTDay_GridPoints_*.csv` |
| `4-MODL_ExtractNDVI250mFromNASA.py` | MODIS MOD13Q1 TIF files + grid points CSV | `NDVI_GridPoints_*.csv` |

Each script:
- Loads the raw satellite data files
- Samples values at each grid point coordinate
- Averages values over the two-year period per season
- Saves seasonal averages to CSV

### Step 3 — Spatial Interpolation (fill missing values)

```
Interpolators_scripts/PrecipitationSpatialInterpolation.py
Interpolators_scripts/Clusterinterpolator_NDVI_LST_Wind_Humidity.py
```

Missing values in extracted data are filled using **inverse-distance weighted (IDW) spatial interpolation** with a KDTree for fast neighbor search. A regional average fallback is used when no spatial neighbors are found within the search radius.

### Step 4 — Merge All Datasets

```
Data extraction scripts/Use_final_datasetmaker_after_individual_extractions/FinalDatasetMaker.py
```

Merges borehole SWL records, climate variables, elevation (DEM), and soil types into a single `Sidama_dataset_for_model.csv` using nearest-coordinate matching (within 10 km threshold). Includes UTM-to-WGS84 coordinate conversion for borehole records.

### Step 5 — Data Diagnostics (optional QC step)

```
DataAnalysisCode/finaldatadiagnostics.py
```

Validates all input datasets: checks shapes, coordinate ranges, spatial overlap, missing values, and data completeness per variable.

### Step 6 — Train Model and Predict

```
Model for water level prediction/Final_Model_For_Prediction_version_2.py
```

Full training and prediction pipeline:
1. Loads and cleans `Sidama_dataset_for_model.csv` (outlier removal using modified Z-score)
2. Engineers features (aggregated climate indices + interaction terms)
3. Scales features with `RobustScaler`
4. Selects top 10 features using Recursive Feature Elimination (RFE)
5. Transforms target (SWL) with Yeo-Johnson `PowerTransformer`
6. Trains `GradientBoostingRegressor` with 5-fold cross-validation
7. Saves all model artifacts to `.pkl` files (`model.pkl`, `scaler.pkl`, `selected_features.pkl`, `target_transformer.pkl`, `soil_mapping.pkl`)
8. Generates predictions for new grid points → `SidamaGridPoints_Predicted.csv`

---

## Features Used in the Model

| Feature | Description |
|---|---|
| `Elevation` | Terrain elevation from SRTM DEM (m) |
| `latitude`, `longitude` | Geographic coordinates |
| `soil_swl_mean` | Mean SWL per soil type (target-encoded) |
| `precip_annual_total` | Sum of seasonal precipitation (mm) |
| `precip_seasonality` | Coefficient of variation of precipitation across seasons |
| `temp_annual_mean` | Mean LST Day across seasons (°C) |
| `humidity_annual_mean` | Mean specific humidity across seasons |
| `elevation_x_precip` | Interaction: elevation × annual precipitation |
| `temp_x_humidity` | Interaction: temperature × humidity |

Climate inputs (pre-aggregation) include per-season values for:
- Precipitation (`Precip_meanCum*`) — 3 seasonal columns
- LST Day (`LSTDay*`) — 3 seasonal columns
- Specific Humidity (`SpecificHumidity_meanCum*`) — 3 seasonal columns
- Wind Speed (`WindSpeedMean*`) — 3 seasonal columns
- NDVI (`NDVI_*`) — 3 seasonal columns

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | `GradientBoostingRegressor` (scikit-learn) |
| n_estimators | 400 |
| learning_rate | 0.03 |
| max_depth | 5 |
| subsample | 0.8 |
| Feature selection | RFE (top 10 features) |
| Feature scaling | `RobustScaler` |
| Target transform | Yeo-Johnson `PowerTransformer` |
| Train/test split | 75% / 25% |
| Cross-validation | 5-fold KFold |
| Outlier removal | Modified Z-score (threshold = 3.5) |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `joblib`
- `rasterio`, `geopandas`, `fiona`, `shapely`
- `netCDF4`, `h5py`
- `matplotlib`

---

## Usage

### 1. Update file paths
All scripts contain hardcoded paths (e.g., `D:\Work Folder\...`). Update the `data_path`, `shapefile_path`, and output directory variables at the top or in the `main()` function of each script before running.

### 2. Run the pipeline in order

```bash
# Step 1: Generate grid points
python "Code (Python)/Data extraction scripts/SidamaGridpointsExtractor/gridpointsforpredictionextractor.py"

# Step 2: Extract variables (run each independently after downloading data)
python "Code (Python)/Data extraction scripts/1-MODL_ExtractPrecipitationNASA.py"
python "Code (Python)/Data extraction scripts/2-MODL_ExtractSpecificHumidityAndWindSpeedFromFLDAS.py"
python "Code (Python)/Data extraction scripts/3-MODL_ExtractLST_MODIS.py"
python "Code (Python)/Data extraction scripts/4-MODL_ExtractNDVI250mFromNASA.py"

# Step 3: Fill missing values
python "Code (Python)/Interpolators_scripts/PrecipitationSpatialInterpolation.py"
python "Code (Python)/Interpolators_scripts/Clusterinterpolator_NDVI_LST_Wind_Humidity.py"

# Step 4: Merge datasets
python "Code (Python)/Data extraction scripts/Use_final_datasetmaker_after_individual_extractions/FinalDatasetMaker.py"

# Step 5 (optional): Run diagnostics
python "Code (Python)/DataAnalysisCode/finaldatadiagnostics.py"

# Step 6: Train model and predict
python "Code (Python)/Model for water level prediction/Final_Model_For_Prediction_version_2.py"
```

---

## Output

| File | Description |
|---|---|
| `SidamaGridPoints_Predicted.csv` | Predicted SWL (m) at each grid point with coordinates |
| `TrainedModel/model.pkl` | Trained GradientBoostingRegressor |
| `TrainedModel/scaler.pkl` | Fitted RobustScaler |
| `TrainedModel/selected_features.pkl` | List of selected feature names |
| `TrainedModel/target_transformer.pkl` | Fitted PowerTransformer for SWL |
| `TrainedModel/soil_mapping.pkl` | Soil type → mean SWL encoding dictionary |

---

## References

- Huffman, G.J., et al. (2019). *GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V06*. GES DISC. https://doi.org/10.5067/GPM/IMERGM/3B-HH.07
- McNally, A. NASA/GSFC/HSL (2018). *FLDAS Noah Land Surface Model L4 Global Monthly 0.1 x 0.1 degree (MERRA-2 and CHIRPS)*. GES DISC. https://disc.gsfc.nasa.gov/datasets/FLDAS_NOAH01_C_GL_M_001/summary
- Wan, Z., Hook, S., Hulley, G. (2015). *MOD11A1 MODIS/Terra Land Surface Temperature/Emissivity Daily L3 Global 1km SIN Grid V006*. NASA EOSDIS LP DAAC. https://doi.org/10.5067/MODIS/MOD11A1.006
- Didan, K. (2015). *MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V006*. NASA EOSDIS LP DAAC. https://doi.org/10.5067/MODIS/MOD13Q1.006
- FAO (2012). *Harmonized World Soil Database v1.2*. FAO/IIASA/ISRIC/ISSCAS/JRC, Rome, Italy and Laxenburg, Austria.
