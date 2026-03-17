import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import re

def extract_ndvi_seasonal_data():
    """Extract NDVI data following original seasonal averaging approach."""
    
    # Define paths
    data_path = "NDVI/"
    point_data_path = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\GridPoints_Predicted_Sidama.csv"
    output_path = "NDVI_extracted2/"
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize flag for file type tracking
    is_ndvi_files = False
    
    # Load point coordinates (same as LST script)
    print("Loading coordinates...")
    point_data = pd.read_csv(point_data_path)
    
    if 'Easting' in point_data.columns and 'Northing' in point_data.columns:
        point_coordinates = point_data[["Easting", "Northing"]].values
        print("Using Easting/Northing columns as Longitude/Latitude")
    elif 'Longitude' in point_data.columns and 'Latitude' in point_data.columns:
        point_coordinates = point_data[["Longitude", "Latitude"]].values
        print("Using Longitude/Latitude columns")
    else:
        raise ValueError("No coordinate columns found")
    
    print(f"Loaded {len(point_coordinates)} points")
    print(f"Coordinate range: Lon({point_coordinates[:,0].min():.3f} to {point_coordinates[:,0].max():.3f}), Lat({point_coordinates[:,1].min():.3f} to {point_coordinates[:,1].max():.3f})")
    
    # Analyze available TIF files and find correct files
    data_dir = Path(data_path)
    all_files = list(data_dir.glob("*.tif"))
    
    print(f"Found {len(all_files)} TIF files")
    if len(all_files) == 0:
        print("No TIF files found!")
        return
    
    # Categorize files by type - look specifically for NDVI files
    file_types = {}
    for file in all_files:
        filename = file.name.lower()
        if '_ndvi_' in filename:  # Look for NDVI specifically
            file_type = 'NDVI'
        elif '_evi_' in filename:
            file_type = 'EVI'  
        elif 'blue_reflectance' in filename:
            file_type = 'Blue_Reflectance'
        elif 'red_reflectance' in filename:
            file_type = 'Red_Reflectance'
        elif 'nir_reflectance' in filename:
            file_type = 'NIR_Reflectance'
        else:
            file_type = 'Other'
        
        if file_type not in file_types:
            file_types[file_type] = []
        file_types[file_type].append(file)
    
    print("Available file types:")
    for file_type, files in file_types.items():
        print(f"  {file_type}: {len(files)} files")
    
    # Use NDVI files specifically
    if 'NDVI' in file_types:
        print(f"✅ Using {len(file_types['NDVI'])} NDVI files")
        files_to_process = file_types['NDVI']
        is_ndvi_files = True
    else:
        print("❌ No NDVI files found!")
        return
    
    # Show sample NDVI filenames
    print("NDVI files to process:")
    for f in files_to_process[:5]:
        print(f"  {f.name}")
    if len(files_to_process) > 5:
        print(f"  ... and {len(files_to_process) - 5} more")
    
    # Extract year and day information from correct files
    file_day_mapping = {}
    years_available = set()
    
    for file in files_to_process:
        # Try multiple patterns to extract year and day
        patterns = [
            r'doy(\d{4})(\d{3})',  # doy2023353
            r'(\d{4})(\d{3})',     # 2023353
            r'A(\d{4})(\d{3})',    # A2023353
            r'(\d{4})\.(\d{3})',   # 2023.353
        ]
        
        day_match = None
        for pattern in patterns:
            day_match = re.search(pattern, file.name)
            if day_match:
                break
        
        if day_match:
            year = day_match.group(1)
            day = int(day_match.group(2))
            
            years_available.add(year)
            if year not in file_day_mapping:
                file_day_mapping[year] = {}
            file_day_mapping[year][day] = file
    
    print(f"Available years: {sorted(years_available)}")
    for year in sorted(years_available):
        if year in file_day_mapping:
            days = sorted(file_day_mapping[year].keys())
            print(f"  {year}: days {min(days)} to {max(days)} ({len(days)} files)")
    
    if not years_available:
        print("Could not extract year/day information from filenames")
        print("Please check filename patterns. Expected patterns: doy2023353, A2023353, etc.")
        return
    
    # Select years to process (prefer 2 consecutive years like original)
    years_to_process = sorted(list(years_available))
    if len(years_to_process) >= 2:
        years = years_to_process[:2]  # Use first 2 years
        print(f"Processing 2 years: {years} (like original)")
    else:
        years = years_to_process
        print(f"Processing {len(years)} year(s): {years}")
    
    # Define seasonal day ranges (following original instructions)
    seasonal_ranges = {
        'big_rains_Jun_Sep': range(152, 274),      # June to September
        'dry_season_Oct_Jan': list(range(274, 366)) + list(range(1, 32)),  # Oct-Dec + Jan
        'small_rains_Feb_May': range(32, 152)      # February to May
    }
    
    print("Seasonal day ranges:")
    for season, days in seasonal_ranges.items():
        if isinstance(days, range):
            print(f"  {season}: days {min(days)} to {max(days)}")
        else:
            print(f"  {season}: days {min(days)} to {max(days)} (spans year boundary)")
    
    # Extract data for each year and season
    all_seasonal_data = {}
    
    for year in years:
        print(f"\nProcessing year {year}...")
        all_seasonal_data[year] = {}
        
        for season_name, day_range in seasonal_ranges.items():
            print(f"  Processing {season_name}...")
            
            season_data = []
            files_found = 0
            
            for day in day_range:
                # Handle year boundary for dry season
                if season_name == 'dry_season_Oct_Jan' and day < 32:
                    # For Jan days, look in next year
                    next_year = str(int(year) + 1)
                    if next_year in file_day_mapping and day in file_day_mapping[next_year]:
                        filename = file_day_mapping[next_year][day]
                        files_found += 1
                    else:
                        continue
                else:
                    # Normal case
                    if day in file_day_mapping[year]:
                        filename = file_day_mapping[year][day]
                        files_found += 1
                    else:
                        continue
                
                        # Extract data from this file
                try:
                    with rasterio.open(str(filename)) as src:
                        # Print file info for first file of first season
                        if season_name == list(seasonal_ranges.keys())[0] and files_found == 1:
                            print(f"    File info for {filename.name}:")
                            print(f"      CRS: {src.crs}")
                            print(f"      Shape: {src.shape}")
                            print(f"      Data type: {src.dtypes[0]}")
                            print(f"      No data value: {src.nodata}")
                            if not is_ndvi_files:
                                print(f"      ⚠️  WARNING: This is NOT an NDVI file!")
                        
                        # Convert coordinates to (lon, lat) tuples
                        coord_pairs = [(coord[0], coord[1]) for coord in point_coordinates]
                        
                        # Sample data at point locations
                        samples = list(src.sample(coord_pairs, indexes=1))
                        
                        # Convert to NDVI values with proper scaling
                        day_data = []
                        for sample in samples:
                            if len(sample) > 0 and sample[0] != src.nodata:
                                value = sample[0]
                                
                                # Apply MODIS NDVI scale factor (0.0001 from metadata)
                                ndvi_value = value * 0.0001
                                
                                day_data.append(ndvi_value)
                            else:
                                day_data.append(np.nan)
                        
                        season_data.append(day_data)
                        
                except Exception as e:
                    print(f"    Error processing {filename.name}: {e}")
            
            print(f"    Found {files_found} files for {season_name}")
            
            if files_found > 0:
                # Convert to numpy array (points x days)
                all_seasonal_data[year][season_name] = np.array(season_data).T
                
                # Show sample statistics for first season
                if season_name == list(seasonal_ranges.keys())[0]:
                    sample_data = np.array(season_data).T
                    valid_values = sample_data[~np.isnan(sample_data)]
                    if len(valid_values) > 0:
                        print(f"    Sample NDVI range: {np.min(valid_values):.3f} to {np.max(valid_values):.3f}")
                        
                        # Check if values are in expected NDVI range
                        if -1 <= np.min(valid_values) and np.max(valid_values) <= 1:
                            print(f"    ✅ NDVI values in expected range [-1, 1]")
                        else:
                            print(f"    ⚠️  NDVI values outside expected range [-1, 1]")
            else:
                print(f"    No files found for {season_name}")
                all_seasonal_data[year][season_name] = np.full((len(point_coordinates), 1), np.nan)
    
    # Calculate seasonal averages (following original logic)
    print("\nCalculating seasonal averages...")
    
    for season_name in seasonal_ranges.keys():
        print(f"Processing {season_name}...")
        
        # Collect data from all years for this season
        all_years_data = []
        for year in years:
            if year in all_seasonal_data and season_name in all_seasonal_data[year]:
                year_data = all_seasonal_data[year][season_name]
                if year_data.size > 0 and not np.all(np.isnan(year_data)):
                    # Average across days within this year
                    with np.errstate(invalid='ignore'):
                        year_seasonal_mean = np.nanmean(year_data, axis=1)
                    
                    if not np.all(np.isnan(year_seasonal_mean)):
                        all_years_data.append(year_seasonal_mean)
        
        if all_years_data:
            # Average across years (like original approach)
            if len(all_years_data) > 1:
                with np.errstate(invalid='ignore'):
                    combined_data = np.nanmean(np.array(all_years_data), axis=0)
                print(f"  Averaged {len(all_years_data)} years")
            else:
                combined_data = all_years_data[0]
                print(f"  Used single year")
            
            # Save seasonal average
            if len(years) > 1:
                year_str = f"{years[0]}-{years[-1]}"
            else:
                year_str = years[0]
            
            output_filename = f"{output_path}/NDVI_GridPoints_{year_str}_{season_name}.csv"
            
            # Save as single column (like original)
            np.savetxt(output_filename, combined_data, delimiter=",")
            
            # Validate NDVI range
            valid_values = combined_data[~np.isnan(combined_data)]
            if len(valid_values) > 0:
                range_str = f"{np.min(valid_values):.3f} to {np.max(valid_values):.3f}"
                if np.min(valid_values) < -1 or np.max(valid_values) > 1:
                    print(f"  ⚠️  Warning: NDVI values outside normal range [-1, 1]")
                    print(f"  💡 Tip: Check if different scaling factor is needed")
                else:
                    print(f"  ✅ NDVI values in normal range [-1, 1]")
            else:
                range_str = "No valid data"
            
            print(f"  Saved: {output_filename}")
            print(f"  Shape: {combined_data.shape}")
            print(f"  Value range: {range_str}")
            print(f"  Valid data points: {len(valid_values)}/{len(combined_data)}")
        else:
            print(f"  No valid data for {season_name}")
    
    print("NDVI seasonal extraction completed!")
    
    if not is_ndvi_files:
        print("\n" + "="*50)
        print("⚠️  IMPORTANT WARNING:")
        print("The extracted data is NOT from NDVI files!")
        print("You processed reflectance/quality files instead.")
        print("For proper vegetation analysis, you need:")
        print("1. NDVI files from MOD13Q1 product")
        print("2. Files with 'NDVI' in the filename")
        print("="*50)

if __name__ == "__main__":
    extract_ndvi_seasonal_data()