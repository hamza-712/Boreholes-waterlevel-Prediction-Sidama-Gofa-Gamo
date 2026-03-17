import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def convert_utm_to_latlon(easting, northing, zone=37, hemisphere='N'):
    """
    Convert UTM coordinates to Latitude/Longitude
    This is a simplified conversion - for production use pyproj library
    """
    # Simplified conversion for Zone 37N (approximate for Ethiopia region)
    # For accurate conversion, use: from pyproj import Proj, transform
    
    # Rough conversion (this is approximate - replace with proper conversion if needed)
    central_meridian = (zone - 1) * 6 - 180 + 3  # Central meridian for zone 37 = 39°E
    
    # Simplified inverse projection (not accurate for all cases)
    x = easting - 500000  # Remove false easting
    y = northing if hemisphere == 'N' else northing - 10000000  # Remove false northing for southern hemisphere
    
    # Very rough approximation - replace with proper UTM to LatLon conversion
    lon = central_meridian + (x / 111320.0)
    lat = y / 110540.0
    
    return lat, lon

def find_nearest_coordinates(target_coords, source_coords, max_distance_km=5.0):
    """
    Find nearest coordinates within a maximum distance threshold
    Returns indices of nearest matches
    """
    # Convert to numpy arrays
    target = np.array(target_coords)
    source = np.array(source_coords)
    
    # Remove any NaN coordinates
    target_valid = ~(np.isnan(target).any(axis=1))
    source_valid = ~(np.isnan(source).any(axis=1))
    
    if not target_valid.any() or not source_valid.any():
        print("Warning: No valid coordinates found!")
        return [None] * len(target)
    
    # Calculate distances in degrees (approximate)
    distances = cdist(target, source, metric='euclidean')
    
    # Convert distance threshold from km to degrees (rough approximation)
    max_distance_degrees = max_distance_km / 111.0  # 1 degree ≈ 111 km
    print(f"Using distance threshold: {max_distance_degrees:.4f} degrees ({max_distance_km} km)")
    
    # Find nearest neighbors
    nearest_indices = []
    successful_matches = 0
    min_distances = []
    
    for i in range(len(target)):
        if not target_valid[i]:
            nearest_indices.append(None)
            continue
            
        min_dist_idx = np.argmin(distances[i])
        min_distance = distances[i, min_dist_idx]
        min_distances.append(min_distance)
        
        if min_distance <= max_distance_degrees:
            nearest_indices.append(min_dist_idx)
            successful_matches += 1
        else:
            nearest_indices.append(None)
    
    # Print debugging info
    if min_distances:
        print(f"Distance statistics (degrees): min={min(min_distances):.4f}, max={max(min_distances):.4f}, mean={np.mean(min_distances):.4f}")
        print(f"Distance statistics (km): min={min(min_distances)*111:.2f}, max={max(min_distances)*111:.2f}, mean={np.mean(min_distances)*111:.2f}")
    
    print(f"Successful matches: {successful_matches}/{len(target)} with threshold {max_distance_km}km")
    
    return nearest_indices

def extract_swl_data(file_path):
    """Extract and clean SWL data from sidama_375_records.csv"""
    print("Loading SWL data...")
    df = pd.read_csv(file_path)
    
    # Convert UTM to Lat/Lon if needed
    if 'X_final_WGS84UTM37N' in df.columns and 'Y_final_WGS84UTM37N' in df.columns:
        print("Converting UTM coordinates to Lat/Lon...")
        latitudes, longitudes = [], []
        for _, row in df.iterrows():
            if pd.notna(row['X_final_WGS84UTM37N']) and pd.notna(row['Y_final_WGS84UTM37N']):
                lat, lon = convert_utm_to_latlon(row['X_final_WGS84UTM37N'], row['Y_final_WGS84UTM37N'])
                latitudes.append(lat)
                longitudes.append(lon)
            else:
                latitudes.append(np.nan)
                longitudes.append(np.nan)
        
        df['latitude'] = latitudes
        df['longitude'] = longitudes
    
    # Clean SWL data
    df['SWL_clean'] = pd.to_numeric(df['SWL'], errors='coerce')
    
    # Select relevant columns
    swl_data = df[['WellID', 'latitude', 'longitude', 'SWL_clean']].copy()
    swl_data = swl_data.dropna(subset=['latitude', 'longitude', 'SWL_clean'])
    
    # Add fid and ID columns
    swl_data['fid'] = range(1, len(swl_data) + 1)
    swl_data['ID'] = swl_data['fid']
    
    print(f"Extracted {len(swl_data)} valid SWL records")
    return swl_data

def load_climate_data(file_path):
    """Load climate data from CleanSidamaForPrediction.xlsx"""
    print("Loading climate data...")
    df = pd.read_excel(file_path)
    
    # Print columns to understand structure
    print("Climate data columns:", df.columns.tolist())
    
    # Handle duplicate column names by renaming them first
    new_columns = []
    column_counts = {}
    
    for col in df.columns:
        if col in column_counts:
            column_counts[col] += 1
            new_col = f"{col}.{column_counts[col]}"
        else:
            column_counts[col] = 0
            new_col = col
        new_columns.append(new_col)
    
    df.columns = new_columns
    print("Renamed columns to handle duplicates:", df.columns.tolist())
    
    # Create mapping for expected column names
    column_mapping = {
        # Wind Speed
        'WindSpeedMeanJunToSep': 'WindSpeedMeanJunToSep06-07',
        'WindSpeedMeanOctToJan': 'WindSpeedMeanOctToJan05-07', 
        'WindSpeedMeanFebToMay': 'WindSpeedMeanFebToMay06-07',
        
        # Precipitation
        'Precipitation Oct-Jan': 'Precip_meanCumOctToJan',
        'Feb-May': 'Precip_meanCumFebToMay',  # First Feb-May is precipitation
        'Jun-Sep': 'Precip_meanCumJunToSep',
        
        # NDVI
        'NDVI-June-Sep': 'NDVI_Jun.Sep_2007',
        'NDVI-dry-Oct-Jan': 'NDVI_Oct.Jan_2006-2007',
        'Feb-May.1': 'NDVI_Feb.May_2007',  # Second Feb-May is NDVI
        
        # LST Day temperatures
        'LST-234-243': 'LSTDayMeanJunToSep06-07',  # Days 234-243 (Jun-Sep)
        'LST-273-324': 'LSTDayMeanOctToJan05-07',  # Days 273-324 (Oct-Jan) 
        'LST-326-356': 'LSTDayMeanFebToMay06-07',  # Days 326-356 (Feb-May)
        
        # LST Night temperatures - we only have one, so we'll use it for all seasons
        # This is a limitation of your source data
        'LST-358-365': 'LSTNightMeanOctToJan05-07', # Days 358-365 (Oct-Jan period)
        
        # Specific Humidity
        'SpecificHumidity_meanCumJunToSep': 'SpecificHumidity_meanCumJunToSep',
        'SpecificHumidity_meanCumOctToJan': 'SpecificHumidity_meanCumOctToJan',
        'SpecificHumidity_meanCumFebToMay': 'SpecificHumidity_meanCumFebToMay'
    }
    
    # Apply column mappings
    df_renamed = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_renamed.columns:
            df_renamed = df_renamed.rename(columns={old_name: new_name})
    
    # CRITICAL: Filter to only records with at least some climate data
    climate_cols = [col for col in df_renamed.columns if col not in ['longitude', 'latitude', 'elevation', 'longitude.1', 'latitude.1']]
    
    # Create a mask for rows that have at least one non-null climate value
    has_climate_data = df_renamed[climate_cols].notna().any(axis=1)
    df_filtered = df_renamed[has_climate_data].copy()
    
    print(f"Original climate records: {len(df_renamed)}")
    print(f"Records with actual climate data: {len(df_filtered)} ({len(df_filtered)/len(df_renamed)*100:.1f}%)")
    
    # Check data availability by variable
    print("Data availability by variable:")
    for col in climate_cols:
        if col in df_filtered.columns:
            non_null = df_filtered[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df_filtered)} ({non_null/len(df_filtered)*100:.1f}%)")
    
    return df_filtered

def load_elevation_data(file_path):
    """Load elevation data from sidama_dem_grid.csv"""
    print("Loading elevation data...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} elevation records")
    return df

def load_soil_data(file_path):
    """Load soil data from soil_groups.csv"""
    print("Loading soil data...")
    df = pd.read_csv(file_path)
    df = df.rename(columns={'soil_group_value': 'SOIL_TYPE'})
    print(f"Loaded {len(df)} soil records")
    return df

def merge_datasets(swl_data, climate_data, elevation_data, soil_data, max_distance_km=10.0):
    """Merge all datasets based on coordinate matching"""
    print(f"\nMerging datasets with distance threshold: {max_distance_km}km...")
    
    # Start with SWL data as base
    result = swl_data.copy()
    
    # Merge climate data
    print("Merging climate data...")
    if 'latitude' in climate_data.columns and 'longitude' in climate_data.columns:
        swl_coords = list(zip(result['latitude'], result['longitude']))
        climate_coords = list(zip(climate_data['latitude'], climate_data['longitude']))
        
        print(f"SWL coordinate range: lat {min(result['latitude']):.3f} to {max(result['latitude']):.3f}, lon {min(result['longitude']):.3f} to {max(result['longitude']):.3f}")
        print(f"Climate coordinate range: lat {min(climate_data['latitude']):.3f} to {max(climate_data['latitude']):.3f}, lon {min(climate_data['longitude']):.3f} to {max(climate_data['longitude']):.3f}")
        
        # Test with first few coordinates
        print(f"Sample SWL coordinates: {swl_coords[:3]}")
        print(f"Sample climate coordinates: {climate_coords[:3]}")
        
        climate_indices = find_nearest_coordinates(swl_coords, climate_coords, max_distance_km)
        
        # Count successful matches
        successful_matches = sum(1 for idx in climate_indices if idx is not None)
        print(f"Successfully matched {successful_matches}/{len(climate_indices)} records")
        
        # Initialize climate columns with NaN
        climate_columns = [col for col in climate_data.columns if col not in ['latitude', 'longitude', 'elevation', 'longitude.1', 'latitude.1']]
        print(f"Climate columns to merge: {climate_columns}")
        
        for col in climate_columns:
            result[col] = np.nan
        
        # Fill matched data
        for i, idx in enumerate(climate_indices):
            if idx is not None:
                for col in climate_columns:
                    if col in climate_data.columns:
                        result.iloc[i, result.columns.get_loc(col)] = climate_data.iloc[idx][col]
        
        # Report data availability after merge
        print("Post-merge climate data availability:")
        for col in climate_columns:
            if col in result.columns:
                non_null = result[col].notna().sum()
                print(f"  {col}: {non_null}/{len(result)} ({non_null/len(result)*100:.1f}%)")
    
    # Merge elevation data (keep same distance threshold)
    print("\nMerging elevation data...")
    swl_coords = list(zip(result['latitude'], result['longitude']))
    elevation_coords = list(zip(elevation_data['latitude'], elevation_data['longitude']))
    
    elevation_indices = find_nearest_coordinates(swl_coords, elevation_coords, max_distance_km)
    
    result['Elevation'] = np.nan
    for i, idx in enumerate(elevation_indices):
        if idx is not None:
            result.iloc[i, result.columns.get_loc('Elevation')] = elevation_data.iloc[idx]['elevation']
    
    # Merge soil data (keep same distance threshold)
    print("\nMerging soil data...")
    soil_coords = list(zip(soil_data['latitude'], soil_data['longitude']))
    
    soil_indices = find_nearest_coordinates(swl_coords, soil_coords, max_distance_km)
    
    result['SOIL_TYPE'] = np.nan
    for i, idx in enumerate(soil_indices):
        if idx is not None:
            result.iloc[i, result.columns.get_loc('SOIL_TYPE')] = soil_data.iloc[idx]['SOIL_TYPE']
    
    return result
    
    # Merge elevation data
    print("Merging elevation data...")
    swl_coords = list(zip(result['latitude'], result['longitude']))
    elevation_coords = list(zip(elevation_data['latitude'], elevation_data['longitude']))
    
    elevation_indices = find_nearest_coordinates(swl_coords, elevation_coords, max_distance_km)
    
    result['Elevation'] = np.nan
    for i, idx in enumerate(elevation_indices):
        if idx is not None:
            result.iloc[i, result.columns.get_loc('Elevation')] = elevation_data.iloc[idx]['elevation']
    
    # Merge soil data
    print("Merging soil data...")
    soil_coords = list(zip(soil_data['latitude'], soil_data['longitude']))
    
    soil_indices = find_nearest_coordinates(swl_coords, soil_coords, max_distance_km)
    
    result['SOIL_TYPE'] = np.nan
    for i, idx in enumerate(soil_indices):
        if idx is not None:
            result.iloc[i, result.columns.get_loc('SOIL_TYPE')] = soil_data.iloc[idx]['SOIL_TYPE']
    
    return result

def create_final_dataset(result_df):
    """Create final dataset with required column order and names"""
    print("Creating final dataset structure...")
    
    # Define required columns in order
    required_columns = [
        'fid', 'ID', 'latitude', 'longitude', 'Elevation', 'SWL_clean',
        'SOIL_TYPE', 'SpecificHumidity_meanCumOctToJan', 'SpecificHumidity_meanCumFebToMay',
        'SpecificHumidity_meanCumJunToSep', 'WindSpeedMeanOctToJan05-07', 'WindSpeedMeanFebToMay06-07',
        'WindSpeedMeanJunToSep06-07', 'LSTDayMeanOctToJan05-07', 'LSTDayMeanFebToMay06-07',
        'LSTDayMeanJunToSep06-07', 'LSTNightMeanOctToJan05-07', 'LSTNightMeanFebToMay06-07',
        'LSTNightMeanJunToSep06-07', 'NDVI_Oct.Jan_2006-2007', 'NDVI_Feb.May_2007',
        'NDVI_Jun.Sep_2007', 'Precip_meanCumOctToJan', 'Precip_meanCumFebToMay',
        'Precip_meanCumJunToSep'
    ]
    
    # Create final dataframe with required columns
    final_df = pd.DataFrame()
    
    for col in required_columns:
        if col == 'SWL_clean':
            final_df['SWL'] = result_df['SWL_clean'] if 'SWL_clean' in result_df.columns else np.nan
        elif col in result_df.columns:
            final_df[col] = result_df[col]
        else:
            final_df[col] = np.nan
            print(f"Warning: Column '{col}' not found in merged data, filled with NaN")
    
    # Handle missing LST night columns: use available night data or day data as fallback
    if 'LSTNightMeanOctToJan05-07' in result_df.columns:
        # We have Oct-Jan night data, use it for other seasons as approximation
        night_oct_jan = result_df['LSTNightMeanOctToJan05-07']
        
        if final_df['LSTNightMeanFebToMay06-07'].isna().all():
            print("Using Oct-Jan night LST for Feb-May night LST (approximation)")
            final_df['LSTNightMeanFebToMay06-07'] = night_oct_jan
        
        if final_df['LSTNightMeanJunToSep06-07'].isna().all():
            print("Using Oct-Jan night LST for Jun-Sep night LST (approximation)")
            final_df['LSTNightMeanJunToSep06-07'] = night_oct_jan
    else:
        # Fallback: use day temperatures as night temperature approximation
        print("Warning: No night LST data found, using day LST as approximation")
        final_df['LSTNightMeanOctToJan05-07'] = final_df.get('LSTDayMeanOctToJan05-07', np.nan)
        final_df['LSTNightMeanFebToMay06-07'] = final_df.get('LSTDayMeanFebToMay06-07', np.nan)
        final_df['LSTNightMeanJunToSep06-07'] = final_df.get('LSTDayMeanJunToSep06-07', np.nan)
    
    return final_df

def main():
    """Main function to execute the data merging process"""
    # File paths - update these with your actual file paths (use raw strings for Windows paths)
    swl_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\sidama_375_records.csv"
    climate_file = r"D:\\Work Folder\\maaz Work Africa water level\\OneDrive_2025-01-06\\SGG zones\\CleanSidamaForPrediction_FAST_interpolated.xlsx"
    elevation_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\sidama_dem_grid.csv"
    soil_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\Submission 1\soil_groups.csv"
    
    try:
        # Load all datasets
        swl_data = extract_swl_data(swl_file)
        climate_data = load_climate_data(climate_file)
        elevation_data = load_elevation_data(elevation_file)
        soil_data = load_soil_data(soil_file)
        
        # Merge datasets
        merged_data = merge_datasets(swl_data, climate_data, elevation_data, soil_data)
        
        # Create final dataset
        final_dataset = create_final_dataset(merged_data)
        
        # Save to CSV
        output_file = "Sidama_dataset_for_model.csv"
        final_dataset.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully created {output_file}")
        print(f"Dataset shape: {final_dataset.shape}")
        print(f"Columns: {list(final_dataset.columns)}")
        
        # Print data quality summary
        print("\n📊 Data Quality Summary:")
        print(f"Total records: {len(final_dataset)}")
        print(f"Complete records (no missing values): {final_dataset.dropna().shape[0]}")
        
        missing_summary = final_dataset.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary) > 0:
            print("\nMissing values by column:")
            for col, count in missing_summary.items():
                print(f"  {col}: {count} ({count/len(final_dataset)*100:.1f}%)")
        
        return final_dataset
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return None

if __name__ == "__main__":
    final_dataset = main()