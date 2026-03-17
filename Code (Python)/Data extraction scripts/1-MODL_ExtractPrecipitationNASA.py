import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask
from shapely.geometry import Point

# Configuration - Update these paths for your system
data_paths = [
    r"D:\Work Folder\maaz Work Africa water level\GPM_3IMERGM_data_apr282024to2025\GPM_3IMERGM_07-20250427_224012",
    r"D:\Work Folder\maaz Work Africa water level\GPM_3IMERGM_data_apr282024to2025\GPM_3IMERGM_07-20250427_223225"
]

coordinates_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\GridPoints_Predicted_Sidama.csv"

# Target years for 2-year seasonal average (based on your available data)
target_years = ["2001", "2002"]  # Using complete consecutive years from your data

print("Starting GPM precipitation extraction...")

# Load coordinates file
print("Loading coordinates...")
point_coordinates = pd.read_csv(coordinates_file)
print(f"Loaded {len(point_coordinates)} coordinate points")
print(f"Available columns: {list(point_coordinates.columns)}")

# Auto-detect longitude and latitude columns
lon_cols = ['longitude', 'Longitude', 'lon', 'Lon', 'long', 'Long', 'Longitute']
lat_cols = ['latitude', 'Latitude', 'lat', 'Lat']

lon_col = None
lat_col = None

for col in lon_cols:
    if col in point_coordinates.columns:
        lon_col = col
        break

for col in lat_cols:
    if col in point_coordinates.columns:
        lat_col = col
        break

if not lon_col or not lat_col:
    print(f"Error: Could not find longitude/latitude columns!")
    print(f"Available columns: {list(point_coordinates.columns)}")
    exit()

print(f"Using longitude column: {lon_col}")
print(f"Using latitude column: {lat_col}")

# Find all HDF5 files in data directories
print("Searching for HDF5 files...")
all_files = []
for data_path in data_paths:
    if os.path.exists(data_path):
        pattern = os.path.join(data_path, "**/*.HDF5")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    else:
        print(f"Warning: Path does not exist: {data_path}")

print(f"Found {len(all_files)} HDF5 files")

# Extract year-month from filenames and filter for target years
target_files = {}
for file_path in all_files:
    filename = os.path.basename(file_path)
    # Extract date from filename (format: 3B-MO.MS.MRG.3IMERG.YYYYMMDD-...)
    if "3B-MO.MS.MRG.3IMERG." in filename:
        date_part = filename.split("3B-MO.MS.MRG.3IMERG.")[1][:8]  # YYYYMMDD
        if len(date_part) == 8:
            year = date_part[:4]
            month = date_part[4:6]
            yrmo = year + month
            
            if year in target_years:
                target_files[yrmo] = file_path

print(f"Found {len(target_files)} files for target years {target_years}")
print(f"Available months: {sorted(target_files.keys())}")

if len(target_files) == 0:
    print("No files found for target years!")
    exit()

# Extract precipitation data for each month
print("Extracting precipitation data...")
mon_prec = []  # List to store monthly precipitation for all points

# Sort months for consistent processing
sorted_months = sorted(target_files.keys())

for yrmo in sorted_months:
    file_path = target_files[yrmo]
    print(f"Processing {yrmo}: {os.path.basename(file_path)}")
    
    try:
        # Load the netCDF file
        nc_data = Dataset(file_path, 'r')
        
        # Read longitude, latitude, and precipitation data
        lon = nc_data.variables['Grid/lon'][:]
        lat = nc_data.variables['Grid/lat'][:]
        prec_array = nc_data.variables['Grid/precipitation'][:]
        fill_value = nc_data.variables['Grid/precipitation']._FillValue
        
        # Replace fill values with NaN
        prec_array = np.where(prec_array == fill_value, np.nan, prec_array)
        
        # Close the netCDF file
        nc_data.close()
        
        # Create raster from precipitation data
        transform = from_bounds(np.min(lon), np.min(lat), np.max(lon), np.max(lat), 
                              len(lon), len(lat))
        
        temp_raster = f'temp_precipitation_{yrmo}.tif'
        with rasterio.open(
            temp_raster,
            'w',
            driver='GTiff',
            height=prec_array.shape[0],
            width=prec_array.shape[1],
            count=1,
            dtype=prec_array.dtype,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=transform,
        ) as dst:
            dst.write(prec_array, 1)
        
        # Extract precipitation values for all coordinate points
        month_values = []
        with rasterio.open(temp_raster) as src:
            for i, row in point_coordinates.iterrows():
                point_lon = row[lon_col]
                point_lat = row[lat_col]
                point = Point(point_lon, point_lat)
                
                try:
                    # Extract value at point location
                    masked_data, _ = mask(src, [point], crop=True, all_touched=True)
                    masked_data = masked_data.flatten()
                    
                    # Get first non-NaN value, or NaN if all are NaN
                    non_nan_values = masked_data[~np.isnan(masked_data)]
                    if len(non_nan_values) > 0:
                        month_values.append(non_nan_values[0])
                    else:
                        month_values.append(np.nan)
                        
                except Exception as e:
                    print(f"Warning: Error extracting value at ({point_lon}, {point_lat}): {e}")
                    month_values.append(np.nan)
        
        mon_prec.append(month_values)
        
        # Clean up temporary raster
        if os.path.exists(temp_raster):
            os.remove(temp_raster)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

# Convert to DataFrame
mon_prec_df = pd.DataFrame(mon_prec).T
mon_prec_df.columns = sorted_months

print(f"Extracted data shape: {mon_prec_df.shape}")
print("Available months:", list(mon_prec_df.columns))

# Save monthly precipitation data
mon_prec_df.to_csv("ExtractedPrecipitationGridPoints.csv", index=False)
print("Saved monthly precipitation data to: ExtractedPrecipitationGridPoints.csv")

# Convert mm/hour to cumulative mm/month
print("Converting to cumulative precipitation...")
days_in_month = {
    '01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30,
    '07': 31, '08': 31, '09': 30, '10': 31, '11': 30, '12': 31
}

# Convert each month: mm/hour * 24 hours * days_in_month
cumulative_df = mon_prec_df.copy()
for col in mon_prec_df.columns:
    if len(col) >= 6:  # YYYYMM format
        month = col[-2:]  # Extract month
        if month in days_in_month:
            days = days_in_month[month]
            cumulative_df[col] = mon_prec_df[col] * 24 * days

# Calculate seasonal averages
print("Calculating seasonal averages...")

# Define seasons
season_months = {
    'Oct-Jan': ['10', '11', '12', '01'],  # Dry season
    'Feb-May': ['02', '03', '04', '05'],  # Small rains  
    'Jun-Sep': ['06', '07', '08', '09']   # Big rains
}

seasonal_results = {}

for season_name, months in season_months.items():
    print(f"Processing {season_name} season...")
    
    seasonal_values = []
    
    for year in target_years:
        year_values = []
        
        for month in months:
            # Handle year transition for Oct-Jan season
            if season_name == 'Oct-Jan' and month in ['01']:
                # January belongs to the next year for Oct-Jan season
                next_year = str(int(year) + 1)
                col_name = next_year + month
            else:
                col_name = year + month
            
            if col_name in cumulative_df.columns:
                year_values.append(cumulative_df[col_name])
                print(f"  Added {col_name}")
        
        if year_values:
            # Calculate mean for this year
            year_mean = pd.concat(year_values, axis=1).mean(axis=1)
            seasonal_values.append(year_mean)
    
    if seasonal_values:
        # Calculate overall seasonal average across years
        seasonal_results[season_name] = pd.concat(seasonal_values, axis=1).mean(axis=1)

# Create final seasonal DataFrame
seasonal_df = pd.DataFrame(seasonal_results)

print(f"Seasonal data shape: {seasonal_df.shape}")
print("Calculated seasons:", list(seasonal_df.columns))

# Display summary statistics
print("\nSeasonal precipitation summary (mm):")
print(seasonal_df.describe())

# Save seasonal averages
seasonal_df.to_csv('SeasonalPrecipitation.csv', index=False)
print("Saved seasonal precipitation to: SeasonalPrecipitation.csv")

print("\n=== Extraction Complete ===")
print("Files created:")
print("- ExtractedPrecipitationGridPoints.csv (monthly data)")
print("- SeasonalPrecipitation.csv (seasonal averages)")
print(f"Processed {len(point_coordinates)} coordinate points")
print(f"Used data from years: {target_years}")