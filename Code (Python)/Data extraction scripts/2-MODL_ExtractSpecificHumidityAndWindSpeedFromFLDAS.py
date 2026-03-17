import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.mask import mask
import glob
from pathlib import Path
import re
import fiona
import shutil

def inspect_netcdf_structure(file_path):
    """Inspect and print the structure of a NetCDF file."""
    print(f"Inspecting file: {file_path}")
    
    try:
        nc_data = nc.Dataset(file_path)
        
        # Print dimensions
        print("\nDimensions:")
        for dim_name, dim in nc_data.dimensions.items():
            print(f"- {dim_name}: {len(dim)}")
        
        # Print variables
        print("\nVariables:")
        for var_name, var in nc_data.variables.items():
            print(f"- {var_name}: {var.shape} - {var.datatype}")
            
            # Print attributes for each variable
            if hasattr(var, 'long_name'):
                print(f"  Long name: {var.long_name}")
            if hasattr(var, 'units'):
                print(f"  Units: {var.units}")
            
            # Show a few more key attributes if they exist
            for attr in ['_FillValue', 'scale_factor', 'add_offset']:
                if hasattr(var, attr):
                    print(f"  {attr}: {getattr(var, attr)}")
        
        nc_data.close()
    except Exception as e:
        print(f"Error inspecting NetCDF file: {str(e)}")

def extract_variable_for_sidama(file_path, variable_name, shapefile_path, output_dir=None):
    """
    Extract a variable from a FLDAS NetCDF file for the Sidama region and save as GeoTIFF.
    
    Args:
        file_path: Path to the NetCDF file
        variable_name: Name of the variable to extract (e.g., 'Qair_f_tavg')
        shapefile_path: Path to the Sidama shapefile
        output_dir: Directory to save outputs (defaults to current directory)
    
    Returns:
        tuple: (file_date, lon, lat, variable_array_sidama, stats)
    """
    if output_dir is None:
        output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract date from filename (e.g., FLDAS_NOAH01_C_GL_M.A202304.001.nc)
    file_name = os.path.basename(file_path)
    # Extract YYYYMM pattern from filename
    date_match = re.search(r'A(\d{6})', file_name)
    
    if date_match:
        date_part = date_match.group(1)  # This will extract the 6-digit date (YYYYMM)
    else:
        # If no date pattern found, try another pattern
        date_match = re.search(r'ANOM(\d{6})', file_name)
        if date_match:
            date_part = date_match.group(1)
        else:
            print(f"Warning: Could not extract date from filename: {file_name}")
            date_part = "000000"  # Default placeholder
    
    print(f"Processing file: {file_name} (Date: {date_part})")
    
    try:
        # Open NetCDF file
        nc_data = nc.Dataset(file_path)
        
        # Check if the requested variable exists
        if variable_name not in nc_data.variables:
            print(f"Warning: Variable {variable_name} not found in {file_name}")
            nc_data.close()
            return None, None, None, None, None
        
        # Extract longitude, latitude, and the requested variable
        # First, check if the coordinates are called 'X' and 'Y' or 'lon' and 'lat'
        if 'X' in nc_data.variables and 'Y' in nc_data.variables:
            lon = nc_data.variables['X'][:]
            lat = nc_data.variables['Y'][:]
        elif 'lon' in nc_data.variables and 'lat' in nc_data.variables:
            lon = nc_data.variables['lon'][:]
            lat = nc_data.variables['lat'][:]
        else:
            print(f"Warning: Could not find longitude and latitude in {file_name}")
            nc_data.close()
            return None, None, None, None, None
        
        # Extract the variable
        var_array = nc_data.variables[variable_name][:]
        
        # Check if the array is multi-dimensional and handle accordingly
        if len(var_array.shape) == 3:
            # If 3D, take the first time slice (assuming it's [time, lat, lon])
            var_array = var_array[0, :, :]
        
        # Get fill value
        fill_value = getattr(nc_data.variables[variable_name], '_FillValue', None)
        
        # Close the file
        nc_data.close()
        
        # Replace fill values with NaN
        if fill_value is not None:
            var_array = np.where(var_array == fill_value, np.nan, var_array)
        
        # Print basic info
        print(f"Data shape: {var_array.shape}")
        print(f"Data range: {np.nanmin(var_array)} to {np.nanmax(var_array)}")
        
        # Create a temporary global GeoTIFF
        temp_global_raster = os.path.join(output_dir, f"{variable_name}_{date_part}_global_temp.tif")
        
        # Calculate transform for the raster
        # FLDAS usually has a regular grid, so we use from_origin
        transform = from_origin(np.min(lon), np.max(lat), (lon[1] - lon[0]), (lat[1] - lat[0]))
        
        with rasterio.open(
            temp_global_raster,
            'w',
            driver='GTiff',
            height=var_array.shape[0],
            width=var_array.shape[1],
            count=1,
            dtype=var_array.dtype,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=transform,
        ) as dst:
            dst.write(var_array, 1)
        
        # Load Sidama shapefile
        try:
            print(f"Loading shapefile: {shapefile_path}")
            sidama = gpd.read_file(shapefile_path)
            print(f"Shapefile CRS: {sidama.crs}")
            
            # Make sure shapefile is in the correct CRS
            if sidama.crs and sidama.crs != 'EPSG:4326':
                sidama = sidama.to_crs('EPSG:4326')
                print("Reprojected shapefile to EPSG:4326")
            
            # Extract shapes from geodataframe
            shapes = [feature['geometry'] for _, feature in sidama.iterrows()]
            
            # Mask the global raster with the Sidama shapefile
            with rasterio.open(temp_global_raster) as src:
                # Get the CRS of the raster
                print(f"Raster CRS: {src.crs}")
                
                # Clip to Sidama region
                try:
                    out_image, out_transform = mask(src, shapes, crop=True)
                    out_meta = src.meta.copy()
                    
                    # Update metadata for the cropped raster
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    
                    # Save the cropped raster
                    sidama_raster = os.path.join(output_dir, f"{variable_name}_{date_part}_sidama.tif")
                    with rasterio.open(sidama_raster, "w", **out_meta) as dest:
                        dest.write(out_image)
                    
                    print(f"Sidama {variable_name} data saved to {sidama_raster}")
                    
                    # Calculate statistics for the Sidama region
                    sidama_data = out_image[0]  # Get the data from the first band
                    sidama_stats = {
                        'min': np.nanmin(sidama_data),
                        'max': np.nanmax(sidama_data),
                        'mean': np.nanmean(sidama_data),
                        'median': np.nanmedian(sidama_data),
                        'std': np.nanstd(sidama_data)
                    }
                    
                    # Visualize the Sidama data
                    plt.figure(figsize=(10, 8))
                    masked_sidama = np.ma.masked_invalid(sidama_data)
                    plt.imshow(masked_sidama, cmap='viridis')
                    plt.colorbar(label=variable_name)
                    plt.title(f"Sidama {variable_name} - {date_part}")
                    plt.savefig(os.path.join(output_dir, f"{variable_name}_{date_part}_sidama.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Remove the temporary global raster
                    os.remove(temp_global_raster)
                    
                    return date_part, lon, lat, sidama_data, sidama_stats
                    
                except Exception as e:
                    print(f"Error masking raster with shapefile: {str(e)}")
                    # Keep the global raster for debug purposes
                    return date_part, lon, lat, var_array, None
                
        except Exception as e:
            print(f"Error processing shapefile {shapefile_path}: {str(e)}")
            # Return the global data if shapefile processing fails
            return date_part, lon, lat, var_array, None
        
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
        return None, None, None, None, None

def generate_points_for_sidama(shapefile_path, spacing=0.01, output_csv=None):
    """
    Generate a grid of points within the Sidama region for later extraction.
    
    Args:
        shapefile_path: Path to the Sidama shapefile
        spacing: Spacing between points in degrees (default: 0.01 degree, about 1 km)
        output_csv: Path to save the generated points (default: None)
        
    Returns:
        GeoDataFrame: Points within the Sidama region
    """
    try:
        # Load Sidama shapefile
        sidama = gpd.read_file(shapefile_path)
        
        # Make sure shapefile is in the correct CRS
        if sidama.crs and sidama.crs != 'EPSG:4326':
            sidama = sidama.to_crs('EPSG:4326')
        
        # Get the bounds of the shapefile
        bounds = sidama.total_bounds  # [minx, miny, maxx, maxy]
        
        # Generate a grid of points
        x = np.arange(bounds[0], bounds[2], spacing)
        y = np.arange(bounds[1], bounds[3], spacing)
        
        # Create all combinations of x and y
        xx, yy = np.meshgrid(x, y)
        
        # Flatten the grid to get a list of points
        grid_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xx.flatten(), yy.flatten()),
            crs='EPSG:4326'
        )
        
        # Keep only points within the Sidama polygon
        points_in_sidama = gpd.sjoin(grid_points, sidama, how='inner', predicate='within')
        
        print(f"Generated {len(points_in_sidama)} points within the Sidama region")
        
        # Add longitude and latitude columns
        points_in_sidama['longitude'] = points_in_sidama.geometry.x
        points_in_sidama['latitude'] = points_in_sidama.geometry.y
        
        # Save the points to a CSV if requested
        if output_csv:
            points_df = pd.DataFrame({
                'longitude': points_in_sidama.geometry.x,
                'latitude': points_in_sidama.geometry.y
            })
            points_df.to_csv(output_csv, index=False)
            print(f"Saved {len(points_df)} points to {output_csv}")
        
        return points_in_sidama
        
    except Exception as e:
        print(f"Error generating points for Sidama: {str(e)}")
        return None

def extract_stats_for_sidama(data_dir, variable_names, shapefile_path, output_dir="sidama_output"):
    """
    Process FLDAS files and extract statistics for the Sidama region.
    
    Args:
        data_dir: Directory containing NetCDF files
        variable_names: List of variable names to extract
        shapefile_path: Path to the Sidama shapefile
        output_dir: Directory to save outputs
        
    Returns:
        dict: Dictionary with results for each variable
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NetCDF files in the directory
    nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {data_dir}")
        return None
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Sort files to process them in chronological order
    nc_files.sort()
    
    # Initialize dictionaries to store results
    results = {var_name: {'dates': [], 'stats': []} for var_name in variable_names}
    
    # Sample the first file to see its structure
    if nc_files:
        inspect_netcdf_structure(nc_files[0])
    
    # Process each file for each variable
    for var_name in variable_names:
        print(f"\nProcessing variable: {var_name}")
        
        for file_path in nc_files:
            # Extract data for Sidama region
            date_part, lon, lat, sidama_data, sidama_stats = extract_variable_for_sidama(
                file_path, var_name, shapefile_path, output_dir
            )
            
            if date_part is None or sidama_stats is None:
                continue
            
            # Store the date and statistics
            results[var_name]['dates'].append(date_part)
            results[var_name]['stats'].append(sidama_stats)
        
        # Convert the statistics to a DataFrame
        if results[var_name]['dates']:
            stats_df = pd.DataFrame(results[var_name]['stats'])
            stats_df['date'] = results[var_name]['dates']
            
            # Rearrange columns to put date first
            cols = ['date'] + [col for col in stats_df.columns if col != 'date']
            stats_df = stats_df[cols]
            
            # Save the statistics
            stats_df.to_csv(os.path.join(output_dir, f"{var_name}_sidama_stats.csv"), index=False)
            print(f"Statistics for {var_name} saved to {os.path.join(output_dir, f'{var_name}_sidama_stats.csv')}")
            
            # Group by seasons and calculate seasonal means
            def get_season(date_str):
                if len(date_str) >= 6:  # Ensure date_str is in format YYYYMM
                    month = int(date_str[4:6])
                    
                    if 6 <= month <= 9:  # June to September
                        return 'JunToSep'
                    elif month >= 10 or month <= 1:  # October to January
                        return 'OctToJan'
                    elif 2 <= month <= 5:  # February to May
                        return 'FebToMay'
                return 'Unknown'
            
            stats_df['season'] = stats_df['date'].apply(get_season)
            
            # Calculate seasonal means
            seasonal_stats = stats_df.groupby('season').mean().reset_index()
            
            # Save the seasonal statistics
            seasonal_stats.to_csv(os.path.join(output_dir, f"{var_name}_sidama_seasonal_stats.csv"), index=False)
            print(f"Seasonal statistics for {var_name} saved to {os.path.join(output_dir, f'{var_name}_sidama_seasonal_stats.csv')}")
            
            # Store the results
            results[var_name]['stats_df'] = stats_df
            results[var_name]['seasonal_stats'] = seasonal_stats
    
    return results

def extract_points_for_sidama(data_dir, variable_names, shapefile_path, output_dir="sidama_output", grid_spacing=0.01):
    """
    Generate a grid of points within Sidama and extract variable values for each point.
    
    Args:
        data_dir: Directory containing NetCDF files
        variable_names: List of variable names to extract
        shapefile_path: Path to the Sidama shapefile
        output_dir: Directory to save outputs
        grid_spacing: Spacing between grid points in degrees
        
    Returns:
        dict: Dictionary with results for each variable
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate grid points for Sidama
    points_csv = os.path.join(output_dir, "sidama_grid_points.csv")
    sidama_points = generate_points_for_sidama(shapefile_path, spacing=grid_spacing, output_csv=points_csv)
    
    if sidama_points is None or len(sidama_points) == 0:
        print("No points generated for Sidama region")
        return None
    
    # Find all NetCDF files in the directory
    nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {data_dir}")
        return None
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Sort files to process them in chronological order
    nc_files.sort()
    
    # Initialize dictionaries to store results
    results = {var_name: {'monthly_values': None, 'seasonal_means': None} for var_name in variable_names}
    
    # Process each variable
    for var_name in variable_names:
        print(f"\nProcessing variable: {var_name}")
        
        # Initialize a dictionary to store the extracted values for each month
        monthly_values = {}
        
        for file_path in nc_files:
            # Extract date from filename
            file_name = os.path.basename(file_path)
            date_match = re.search(r'A(\d{6})', file_name)
            
            if date_match:
                date_str = date_match.group(1)
            else:
                date_match = re.search(r'ANOM(\d{6})', file_name)
                if date_match:
                    date_str = date_match.group(1)
                else:
                    print(f"Warning: Could not extract date from filename: {file_name}")
                    continue
            
            print(f"Processing {var_name} for date: {date_str}")
            
            # Create a temporary global GeoTIFF for this month
            try:
                # Open NetCDF file
                nc_data = nc.Dataset(file_path)
                
                # Check if the variable exists
                if var_name not in nc_data.variables:
                    print(f"Warning: Variable {var_name} not found in {file_name}")
                    nc_data.close()
                    continue
                
                # Extract coordinates and data
                if 'X' in nc_data.variables and 'Y' in nc_data.variables:
                    lon = nc_data.variables['X'][:]
                    lat = nc_data.variables['Y'][:]
                elif 'lon' in nc_data.variables and 'lat' in nc_data.variables:
                    lon = nc_data.variables['lon'][:]
                    lat = nc_data.variables['lat'][:]
                else:
                    print(f"Warning: Could not find longitude and latitude in {file_name}")
                    nc_data.close()
                    continue
                
                # Extract the variable
                var_array = nc_data.variables[var_name][:]
                
                # Check if the array is multi-dimensional
                if len(var_array.shape) == 3:
                    var_array = var_array[0, :, :]
                
                # Get fill value
                fill_value = getattr(nc_data.variables[var_name], '_FillValue', None)
                
                # Close the NetCDF file
                nc_data.close()
                
                # Replace fill values with NaN
                if fill_value is not None:
                    var_array = np.where(var_array == fill_value, np.nan, var_array)
                
                # Create a temporary GeoTIFF
                temp_raster = os.path.join(output_dir, f"{var_name}_{date_str}_temp.tif")
                
                # Calculate transform
                transform = from_origin(np.min(lon), np.max(lat), (lon[1] - lon[0]), (lat[1] - lat[0]))
                
                with rasterio.open(
                    temp_raster,
                    'w',
                    driver='GTiff',
                    height=var_array.shape[0],
                    width=var_array.shape[1],
                    count=1,
                    dtype=var_array.dtype,
                    crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                    transform=transform,
                ) as dst:
                    dst.write(var_array, 1)
                
                # Extract values for each point
                point_values = []
                
                with rasterio.open(temp_raster) as src:
                    for idx, row in sidama_points.iterrows():
                        lon, lat = row['longitude'], row['latitude']
                        
                        try:
                            # Sample value at point
                            value = list(src.sample([(lon, lat)]))[0][0]
                        except Exception as e:
                            print(f"Error sampling point ({lon}, {lat}): {str(e)}")
                            value = np.nan
                        
                        point_values.append(value)
                
                # Store the values for this month
                monthly_values[date_str] = point_values
                
                # Remove the temporary raster
                os.remove(temp_raster)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Convert the monthly values to a DataFrame
        if monthly_values:
            monthly_df = pd.DataFrame(monthly_values)
            
            # Add coordinates to the DataFrame
            monthly_df['longitude'] = sidama_points['longitude'].values
            monthly_df['latitude'] = sidama_points['latitude'].values
            
            # Save the monthly values
            monthly_df.to_csv(os.path.join(output_dir, f"{var_name}_sidama_monthly_values.csv"), index=False)
            print(f"Monthly values for {var_name} saved to {os.path.join(output_dir, f'{var_name}_sidama_monthly_values.csv')}")
            
            # Calculate seasonal means
            # Initialize dictionaries for seasonal grouping
            big_rains_months = {}
            dry_season_months = {}
            small_rains_months = {}
            
            for date_str in monthly_df.columns:
                if len(date_str) >= 6 and date_str.isdigit():  # Check if it's a date column
                    month = int(date_str[4:6])
                    
                    if 6 <= month <= 9:  # June to September
                        big_rains_months[date_str] = monthly_df[date_str]
                    elif month >= 10 or month <= 1:  # October to January
                        dry_season_months[date_str] = monthly_df[date_str]
                    elif 2 <= month <= 5:  # February to May
                        small_rains_months[date_str] = monthly_df[date_str]
            
            # Prepare column names based on variable
            if var_name == 'Qair_f_tavg':
                col_prefix = 'SpecificHumidity_meanCum'
            elif var_name == 'Wind_f_tavg':
                col_prefix = 'WindSpeedMean'
            else:
                col_prefix = f"{var_name}_mean"
            
            # Calculate seasonal means
            seasonal_means = pd.DataFrame()
            
            if big_rains_months:
                seasonal_means[f'{col_prefix}JunToSep'] = pd.DataFrame(big_rains_months).mean(axis=1)
            
            if dry_season_months:
                seasonal_means[f'{col_prefix}OctToJan'] = pd.DataFrame(dry_season_months).mean(axis=1)
            
            if small_rains_months:
                seasonal_means[f'{col_prefix}FebToMay'] = pd.DataFrame(small_rains_months).mean(axis=1)
            
            # Add coordinates
            seasonal_means['longitude'] = monthly_df['longitude']
            seasonal_means['latitude'] = monthly_df['latitude']
            
            # Save the seasonal means
            seasonal_means.to_csv(os.path.join(output_dir, f"{var_name}_sidama_seasonal_means.csv"), index=False)
            print(f"Seasonal means for {var_name} saved to {os.path.join(output_dir, f'{var_name}_sidama_seasonal_means.csv')}")
            
            # Store the results
            results[var_name]['monthly_values'] = monthly_df
            results[var_name]['seasonal_means'] = seasonal_means
    
    return results

def extract_sidama_shapefile(zip_file_path, output_dir="sidama_shapefile"):
    """
    Extract Sidama shapefile from a zip file.
    
    Args:
        zip_file_path: Path to the zip file containing the shapefile
        output_dir: Directory to extract the shapefile
        
    Returns:
        str: Path to the extracted .shp file
    """
    import zipfile
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all files from the zip
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Find the .shp file
    shp_files = glob.glob(os.path.join(output_dir, "*.shp"))
    
    if not shp_files:
        print(f"No .shp file found in {zip_file_path}")
        return None
    
    # Return the path to the first .shp file
    return shp_files[0]

def cleanup_temp_files(output_dir, pattern="*global_temp.tif"):
    """
    Cleanup all temporary files matching the specified pattern in the output directory.
    
    Args:
        output_dir: Directory to clean up
        pattern: File pattern to match (default: "*global_temp.tif")
    """
    import glob
    import os
    
    # Find all files matching the pattern
    temp_files = glob.glob(os.path.join(output_dir, pattern))
    
    if temp_files:
        print(f"\nCleaning up {len(temp_files)} temporary files...")
        
        # Loop through files and delete them
        for file_path in temp_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to delete {os.path.basename(file_path)}: {str(e)}")
    else:
        print("\nNo temporary files found to clean up.")

def main():
    # Set paths for your data
    fldas_dir = "D:/Work Folder/maaz Work Africa water level/GPM_3IMERGM_data_apr282024to2025/FLDAS_NOAH01_C_GL_M_001-20250427_230046"
    fldas_anom_dir = "D:/Work Folder/maaz Work Africa water level/GPM_3IMERGM_data_apr282024to2025/FLDAS_NOAH01_C_GL_MA_001-20250427_230112"
    
    # Sidama shapefile zip (update this to your actual path)
    sidama_zip = "Sidama.zip"  # Update this with your actual zip file path
    
    # Variables to extract
    variables = ['Qair_f_tavg', 'Wind_f_tavg']
    
    # Output directories
    output_dir_main = "sidama_fldas_output"
    output_dir_anom = "sidama_fldas_anom_output"
    
    # # Extract the Sidama shapefile
    # if os.path.exists(sidama_zip):
    #     shapefile_path = extract_sidama_shapefile(sidama_zip)
    #     if shapefile_path is None:
    #         print("Failed to extract Sidama shapefile")
    #         return
    # else:
    # If zip file doesn't exist, look for the shapefile directly
    shapefile_path = "D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\Sidama\Sidama.shp"  # Update this with your actual shapefile path
    if not os.path.exists(shapefile_path):
        print(f"Sidama shapefile not found at {shapefile_path}")
        return
    
    print(f"Using Sidama shapefile: {shapefile_path}")
    
    # Process regular FLDAS data for Sidama region
    print("\n=== Processing FLDAS data for Sidama region ===")
    if os.path.exists(fldas_dir):
        # Extract statistics for the entire Sidama region
        stats_results = extract_stats_for_sidama(fldas_dir, variables, shapefile_path, output_dir_main)
        
        # Extract data for grid points within Sidama
        grid_results = extract_points_for_sidama(fldas_dir, variables, shapefile_path, output_dir_main, grid_spacing=0.01)
    else:
        print(f"Directory not found: {fldas_dir}")
    
    # Process FLDAS anomaly data for Sidama region
    print("\n=== Processing FLDAS anomaly data for Sidama region ===")
    if os.path.exists(fldas_anom_dir):
        # Extract statistics for the entire Sidama region
        anom_stats_results = extract_stats_for_sidama(fldas_anom_dir, variables, shapefile_path, output_dir_anom)
        
        # Extract data for grid points within Sidama
        anom_grid_results = extract_points_for_sidama(fldas_anom_dir, variables, shapefile_path, output_dir_anom, grid_spacing=0.01)
    else:
        print(f"Directory not found: {fldas_anom_dir}")
    
    print("\nProcessing complete!")
def main2():
     # Output directories
    output_dir_main = "sidama_fldas_output"
    output_dir_anom = "sidama_fldas_anom_output"
    
    cleanup_temp_files(output_dir_main)
    cleanup_temp_files(output_dir_anom)
if __name__ == "__main__":
    #main()
    main2()
