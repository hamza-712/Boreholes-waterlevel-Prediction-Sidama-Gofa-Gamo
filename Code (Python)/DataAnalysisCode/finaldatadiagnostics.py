import pandas as pd
import numpy as np

def diagnose_datasets():
    """Diagnose all datasets to understand their structure and data quality"""
    
    # File paths
    swl_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\sidama_375_records.csv"
    climate_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\CleanSidamaForPrediction.xlsx"
    elevation_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\sidama_dem_grid.csv"
    soil_file = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\Submission 1\soil_groups.csv"
    
    print("="*80)
    print("DATASET DIAGNOSTICS")
    print("="*80)
    
    # 1. SWL Data
    print("\n1. SWL DATA (sidama_375_records.csv)")
    print("-" * 50)
    try:
        swl_df = pd.read_csv(swl_file)
        print(f"Shape: {swl_df.shape}")
        print(f"Columns: {list(swl_df.columns)}")
        print(f"Sample coordinates:")
        if 'X_final_WGS84UTM37N' in swl_df.columns:
            print(f"  UTM X range: {swl_df['X_final_WGS84UTM37N'].min():.0f} to {swl_df['X_final_WGS84UTM37N'].max():.0f}")
            print(f"  UTM Y range: {swl_df['Y_final_WGS84UTM37N'].min():.0f} to {swl_df['Y_final_WGS84UTM37N'].max():.0f}")
        print(f"SWL column info:")
        print(f"  Unique values: {swl_df['SWL'].nunique()}")
        print(f"  Missing values: {swl_df['SWL'].isna().sum()}")
        print(f"  Data types: {swl_df['SWL'].dtype}")
        print(f"  Sample values: {swl_df['SWL'].head().tolist()}")
    except Exception as e:
        print(f"Error loading SWL data: {e}")
    
    # 2. Climate Data
    print("\n2. CLIMATE DATA (CleanSidamaForPrediction.xlsx)")
    print("-" * 50)
    try:
        climate_df = pd.read_excel(climate_file)
        print(f"Shape: {climate_df.shape}")
        print(f"Columns: {list(climate_df.columns)}")
        
        # Check coordinate columns
        coord_cols = ['longitude', 'latitude', 'longitude.1', 'latitude.1']
        for col in coord_cols:
            if col in climate_df.columns:
                print(f"{col}: {climate_df[col].min():.3f} to {climate_df[col].max():.3f}")
        
        # Check data completeness for each climate variable
        climate_vars = [col for col in climate_df.columns if col not in coord_cols + ['elevation']]
        print(f"\nData completeness for climate variables:")
        for var in climate_vars:
            non_null_count = climate_df[var].notna().sum()
            total_count = len(climate_df)
            percentage = (non_null_count / total_count) * 100
            print(f"  {var}: {non_null_count}/{total_count} ({percentage:.1f}%) non-null")
            
            # Show sample values if any exist
            if non_null_count > 0:
                sample_vals = climate_df[var].dropna().head(3).tolist()
                print(f"    Sample values: {sample_vals}")
    except Exception as e:
        print(f"Error loading climate data: {e}")
    
    # 3. Elevation Data
    print("\n3. ELEVATION DATA (sidama_dem_grid.csv)")
    print("-" * 50)
    try:
        elev_df = pd.read_csv(elevation_file)
        print(f"Shape: {elev_df.shape}")
        print(f"Columns: {list(elev_df.columns)}")
        print(f"Coordinate ranges:")
        print(f"  Longitude: {elev_df['longitude'].min():.3f} to {elev_df['longitude'].max():.3f}")
        print(f"  Latitude: {elev_df['latitude'].min():.3f} to {elev_df['latitude'].max():.3f}")
        print(f"  Elevation: {elev_df['elevation'].min():.1f} to {elev_df['elevation'].max():.1f}")
        print(f"Missing elevation values: {elev_df['elevation'].isna().sum()}")
    except Exception as e:
        print(f"Error loading elevation data: {e}")
    
    # 4. Soil Data
    print("\n4. SOIL DATA (soil_groups.csv)")
    print("-" * 50)
    try:
        soil_df = pd.read_csv(soil_file)
        print(f"Shape: {soil_df.shape}")
        print(f"Columns: {list(soil_df.columns)}")
        print(f"Coordinate ranges:")
        print(f"  Longitude: {soil_df['longitude'].min():.3f} to {soil_df['longitude'].max():.3f}")
        print(f"  Latitude: {soil_df['latitude'].min():.3f} to {soil_df['latitude'].max():.3f}")
        print(f"Soil types: {soil_df['soil_group_value'].nunique()} unique")
        print(f"Sample soil types: {soil_df['soil_group_value'].value_counts().head().to_dict()}")
    except Exception as e:
        print(f"Error loading soil data: {e}")
    
    # 5. Coordinate Overlap Analysis
    print("\n5. COORDINATE OVERLAP ANALYSIS")
    print("-" * 50)
    try:
        # Convert SWL UTM to approximate lat/lon for comparison
        swl_df = pd.read_csv(swl_file)
        climate_df = pd.read_excel(climate_file)
        
        # Simple UTM to lat/lon conversion (Zone 37N)
        def utm_to_latlon_simple(easting, northing):
            central_meridian = 39  # Zone 37 central meridian
            x = easting - 500000
            y = northing
            lon = central_meridian + (x / 111320.0)
            lat = y / 110540.0
            return lat, lon
        
        swl_lats, swl_lons = [], []
        for _, row in swl_df.iterrows():
            if pd.notna(row['X_final_WGS84UTM37N']) and pd.notna(row['Y_final_WGS84UTM37N']):
                lat, lon = utm_to_latlon_simple(row['X_final_WGS84UTM37N'], row['Y_final_WGS84UTM37N'])
                swl_lats.append(lat)
                swl_lons.append(lon)
        
        print(f"SWL coordinates (converted): lat {min(swl_lats):.3f} to {max(swl_lats):.3f}, lon {min(swl_lons):.3f} to {max(swl_lons):.3f}")
        print(f"Climate coordinates: lat {climate_df['latitude'].min():.3f} to {climate_df['latitude'].max():.3f}, lon {climate_df['longitude'].min():.3f} to {climate_df['longitude'].max():.3f}")
        
        # Check if ranges overlap
        lat_overlap = not (max(swl_lats) < climate_df['latitude'].min() or min(swl_lats) > climate_df['latitude'].max())
        lon_overlap = not (max(swl_lons) < climate_df['longitude'].min() or min(swl_lons) > climate_df['longitude'].max())
        
        print(f"Latitude overlap: {'YES' if lat_overlap else 'NO'}")
        print(f"Longitude overlap: {'YES' if lon_overlap else 'NO'}")
        
    except Exception as e:
        print(f"Error in overlap analysis: {e}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    diagnose_datasets()