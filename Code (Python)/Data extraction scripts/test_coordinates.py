import rasterio
import numpy as np
import pandas as pd
import os
from pathlib import Path
from rasterio.crs import CRS
from rasterio.warp import transform

def test_coordinate_loading():
    """Test loading and converting coordinates."""
    
    point_data_path = "GridPoints_Predicted_Sidama.csv"
    
    try:
        df = pd.read_csv(point_data_path)
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Check for coordinate columns
        columns = df.columns.str.lower()
        
        if 'easting' in columns and 'northing' in columns:
            east_col = df.columns[columns == 'easting'][0]
            north_col = df.columns[columns == 'northing'][0]
            
            easting = df[east_col].values
            northing = df[north_col].values
            
            print(f"\nEasting/Northing columns found:")
            print(f"Easting range: {easting.min():.6f} to {easting.max():.6f}")
            print(f"Northing range: {northing.min():.6f} to {northing.max():.6f}")
            
            # Check if these are actually geographic coordinates (common mislabeling)
            if 30 <= easting.min() <= 50 and 0 <= northing.min() <= 20:
                print("✓ These appear to be GEOGRAPHIC coordinates (Longitude/Latitude) mislabeled as Easting/Northing!")
                print("✓ No conversion needed - using directly as Longitude/Latitude")
                lons, lats = easting, northing
                
            elif 200000 <= easting.min() <= 800000 and 400000 <= northing.min() <= 1500000:
                print("These appear to be UTM projected coordinates")
                input_crs = CRS.from_epsg(32637)  # UTM Zone 37N
                print(f"Guessed CRS: UTM Zone 37N (EPSG:32637)")
                
                # Convert to geographic coordinates
                lons, lats = transform(input_crs, CRS.from_epsg(4326), easting, northing)
                
                print(f"\nConverted to geographic coordinates:")
                print(f"Longitude range: {lons.min():.6f} to {lons.max():.6f}")
                print(f"Latitude range: {lats.min():.6f} to {lats.max():.6f}")
                
            else:
                print("⚠ Coordinate values don't match expected ranges")
                print("Treating as UTM and attempting conversion...")
                input_crs = CRS.from_epsg(32637)  # Default
                lons, lats = transform(input_crs, CRS.from_epsg(4326), easting, northing)
            
            # Final validation
            if 30 <= lons.min() <= 50 and 0 <= lats.min() <= 20:
                print("✓ Final coordinates look reasonable for Ethiopia/Africa region")
            else:
                print("⚠ Final coordinates might not be correct - check CRS")
                print("Common Ethiopian UTM zones: 32636 (Zone 36N), 32637 (Zone 37N), 32638 (Zone 38N)")
                
        elif 'longitude' in columns and 'latitude' in columns:
            print("Geographic coordinates found - no conversion needed")
            
        else:
            print("❌ No recognizable coordinate columns found")
            print("Expected: 'Easting'/'Northing' or 'Longitude'/'Latitude'")
            
    except Exception as e:
        print(f"Error: {e}")

def debug_everything():
    """Test loading and converting coordinates."""
    
    point_data_path = "GridPoints_Predicted_Sidama.csv"
    
    try:
        df = pd.read_csv(point_data_path)
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Check for coordinate columns
        columns = df.columns.str.lower()
        
        if 'easting' in columns and 'northing' in columns:
            east_col = df.columns[columns == 'easting'][0]
            north_col = df.columns[columns == 'northing'][0]
            
            easting = df[east_col].values
            northing = df[north_col].values
            
            print(f"\nEasting/Northing columns found:")
            print(f"Easting range: {easting.min():.6f} to {easting.max():.6f}")
            print(f"Northing range: {northing.min():.6f} to {northing.max():.6f}")
            
            # Check if these are actually geographic coordinates (common mislabeling)
            if 30 <= easting.min() <= 50 and 0 <= northing.min() <= 20:
                print("✓ These appear to be GEOGRAPHIC coordinates (Longitude/Latitude) mislabeled as Easting/Northing!")
                print("✓ No conversion needed - using directly as Longitude/Latitude")
                lons, lats = easting, northing
                
            elif 200000 <= easting.min() <= 800000 and 400000 <= northing.min() <= 1500000:
                print("These appear to be UTM projected coordinates")
                input_crs = CRS.from_epsg(32637)  # UTM Zone 37N
                print(f"Guessed CRS: UTM Zone 37N (EPSG:32637)")
                
                # Convert to geographic coordinates
                lons, lats = transform(input_crs, CRS.from_epsg(4326), easting, northing)
                
                print(f"\nConverted to geographic coordinates:")
                print(f"Longitude range: {lons.min():.6f} to {lons.max():.6f}")
                print(f"Latitude range: {lats.min():.6f} to {lats.max():.6f}")
                
            else:
                print("⚠ Coordinate values don't match expected ranges")
                print("Treating as UTM and attempting conversion...")
                input_crs = CRS.from_epsg(32637)  # Default
                lons, lats = transform(input_crs, CRS.from_epsg(4326), easting, northing)
            
            # Final validation
            if 30 <= lons.min() <= 50 and 0 <= lats.min() <= 20:
                print("✓ Final coordinates look reasonable for Ethiopia/Africa region")
            else:
                print("⚠ Final coordinates might not be correct - check CRS")
                print("Common Ethiopian UTM zones: 32636 (Zone 36N), 32637 (Zone 37N), 32638 (Zone 38N)")
                
        elif 'longitude' in columns and 'latitude' in columns:
            print("Geographic coordinates found - no conversion needed")
            
        else:
            print("❌ No recognizable coordinate columns found")
            print("Expected: 'Easting'/'Northing' or 'Longitude'/'Latitude'")
            
    except Exception as e:
        print(f"Error: {e}")

def test_single_file_extraction():
    """Test extraction from a single file to debug the process."""
    
    # First test coordinate loading
    print("=== Testing Coordinate Loading ===")
    test_coordinate_loading()
    
    print("\n=== Testing File Access ===")
    
    # Configuration - use forward slashes or raw strings
    data_path = "LST/LST"  # Changed from "LST\LST"
    
    # Check if directory exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please check the path. Current directory contents:")
        current_dir = Path(".")
        print([f.name for f in current_dir.iterdir()])
        return
    
    print(f"Data directory found: {data_dir}")
    print("Directory contents:")
    files = list(data_dir.glob("*"))
    for f in files[:10]:  # Show first 10 files
        print(f"  {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    # Test with any available HDF files
    hdf_files = list(data_dir.glob("*.hdf"))
    if not hdf_files:
        print("No .hdf files found in directory")
        return
    
    test_file = hdf_files[0]  # Use first available HDF file
    print(f"\nTesting file: {test_file}")
    
    try:
        with rasterio.open(str(test_file)) as src:
            print(f"File CRS: {src.crs}")
            print(f"File bounds: {src.bounds}")
            print(f"File shape: {src.shape}")
            print(f"Number of bands: {src.count}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"No data value: {src.nodata}")
            
    except Exception as e:
        print(f"Error processing {test_file}: {e}")

def debug_everything():
    """Test loading and converting coordinates plus file access."""
    
    print("=== Testing Coordinate Loading ===")
    test_coordinate_loading()
    
    print("\n=== Testing File Access ===")
    
    # Configuration - use forward slashes or raw strings
    data_path = "LST/LST"  # Changed from "LST\LST"
    
    # Check if directory exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please check the path. Current directory contents:")
        current_dir = Path(".")
        print([f.name for f in current_dir.iterdir()])
        return
    
    print(f"Data directory found: {data_dir}")
    print("Directory contents:")
    files = list(data_dir.glob("*"))
    for f in files[:10]:  # Show first 10 files
        print(f"  {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    # Test with any available HDF files
    hdf_files = list(data_dir.glob("*.hdf"))
    if not hdf_files:
        print("No .hdf files found in directory")
        return
    
    test_file = hdf_files[0]  # Use first available HDF file
    print(f"\nTesting file: {test_file}")
    
    try:
        with rasterio.open(str(test_file)) as src:
            print(f"File CRS: {src.crs}")
            print(f"File bounds: {src.bounds}")
            print(f"File shape: {src.shape}")
            print(f"Number of bands: {src.count}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"No data value: {src.nodata}")
            
    except Exception as e:
        print(f"Error processing {test_file}: {e}")

if __name__ == "__main__":
    debug_everything()