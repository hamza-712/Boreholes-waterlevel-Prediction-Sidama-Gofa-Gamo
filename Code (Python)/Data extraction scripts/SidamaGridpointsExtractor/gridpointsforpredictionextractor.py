import pandas as pd
import numpy as np
from pyproj import Transformer
import os

def create_sidama_grid_points(borehole_csv, output_csv, grid_spacing=0.01):
    """
    Create a grid of points covering the Sidama area for prediction.
    
    Args:
        borehole_csv: Path to CleanSidamaBoreholes.csv file
        output_csv: Path to save the grid points data
        grid_spacing: Spacing between grid points in degrees
    """
    print("Creating grid points for Sidama region...")
    
    # Load borehole data
    boreholes = pd.read_csv(borehole_csv)
    print(f"Loaded {len(boreholes)} boreholes from {borehole_csv}")
    
    # Get bounding box of the study area
    min_lon = boreholes['Longitute'].min() - 0.05
    max_lon = boreholes['Longitute'].max() + 0.05
    min_lat = boreholes['Latitude'].min() - 0.05
    max_lat = boreholes['Latitude'].max() + 0.05
    
    print(f"Study area bounding box: {min_lon:.2f}, {min_lat:.2f} to {max_lon:.2f}, {max_lat:.2f}")
    
    # Create grid
    lon_range = np.arange(min_lon, max_lon, grid_spacing)
    lat_range = np.arange(min_lat, max_lat, grid_spacing)
    
    # Create all combinations of lat/lon
    grid_points = []
    grid_id = 1
    
    for lon in lon_range:
        for lat in lat_range:
            grid_points.append({
                'Borehole_ID': f'GP_{grid_id}', 
                'Easting': lon,  # Using lon as Easting to match the model's expectation
                'Northing': lat,  # Using lat as Northing to match the model's expectation
                'Elevation': 0    # We'll update this later
            })
            grid_id += 1
    
    # Convert to DataFrame
    grid_df = pd.DataFrame(grid_points)
    
    # Add required columns to match the structure expected by the model
    # Assign the most common soil type from the boreholes
    most_common_soil = boreholes['SOIL_TYPE'].mode()[0]
    grid_df['SOIL_TYPE'] = most_common_soil
    
    # Copy environmental data columns (average values)
    env_columns = [col for col in boreholes.columns if col not in 
                  ['fid', 'ID', 'Easting.m.', 'Nothing.m.', 'Latitude', 'Longitute', 'Elevation', 'SWL', 'SOIL_TYPE']]
    
    for col in env_columns:
        grid_df[col] = boreholes[col].mean()
    
    # Calculate number of grid points
    print(f"Created {len(grid_df)} grid points")
    
    # Save to CSV
    grid_df.to_csv(output_csv, index=False)
    print(f"Saved grid points to {output_csv}")
    
    return grid_df

if __name__ == "__main__":
    # Set paths
    borehole_csv = r"C:\Users\Hp\Downloads\CleanSidamaBoreholes.csv"
    output_csv = r"D:\Work Folder\maaz Work Africa water level\OneDrive_2025-01-06\SGG zones\GridPointsDataForPrediction.csv"

    
    # Create grid points
    grid_df = create_sidama_grid_points(borehole_csv, output_csv)