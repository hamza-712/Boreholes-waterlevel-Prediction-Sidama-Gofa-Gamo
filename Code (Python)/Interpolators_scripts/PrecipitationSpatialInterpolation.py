import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate Euclidean distance between two points in degrees"""
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    return np.sqrt(dlat * dlat + dlon * dlon)

def find_nearest_neighbors(target_lat, target_lon, valid_data, max_distance_deg=0.1, k=5):
    """
    Find k nearest neighbors within max_distance for spatial interpolation
    
    Parameters:
    - target_lat, target_lon: coordinates of point to fill
    - valid_data: DataFrame with valid precipitation data
    - max_distance_deg: maximum distance in degrees (~10km)
    - k: number of nearest neighbors to use
    
    Returns:
    - List of neighbor records with distances
    """
    neighbors = []
    
    for idx, row in valid_data.iterrows():
        distance = calculate_distance(target_lat, target_lon, row['latitude'], row['longitude'])
        if distance <= max_distance_deg:
            neighbors.append({
                'distance': distance,
                'data': row,
                'index': idx
            })
    
    # Sort by distance and return top k
    neighbors.sort(key=lambda x: x['distance'])
    return neighbors[:k]

def spatial_interpolation(target_lat, target_lon, valid_data, columns, max_distance_deg=0.1, k=5):
    """
    Perform inverse distance weighted interpolation for precipitation values
    
    Parameters:
    - target_lat, target_lon: coordinates of point to fill
    - valid_data: DataFrame with valid precipitation data
    - columns: list of precipitation column names to interpolate
    - max_distance_deg: maximum distance for neighbors
    - k: number of neighbors to use
    
    Returns:
    - Dictionary with interpolated values for each column
    """
    neighbors = find_nearest_neighbors(target_lat, target_lon, valid_data, max_distance_deg, k)
    
    if len(neighbors) == 0:
        return None  # No neighbors found
    
    # Perform inverse distance weighting
    interpolated_values = {}
    
    for col in columns:
        if col in valid_data.columns:
            # Get values and distances for this column
            values = []
            distances = []
            
            for neighbor in neighbors:
                val = neighbor['data'][col]
                if pd.notna(val) and val > 0:  # Valid precipitation value
                    values.append(val)
                    # Add small epsilon to avoid division by zero for very close points
                    distances.append(max(neighbor['distance'], 1e-6))
            
            if len(values) > 0:
                # Inverse distance weighting
                weights = [1/d for d in distances]
                total_weight = sum(weights)
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                interpolated_values[col] = weighted_sum / total_weight
            else:
                interpolated_values[col] = None
    
    return interpolated_values

def fill_precipitation_data(file_path, output_path=None):
    """
    Fill missing precipitation data using spatial interpolation
    
    Parameters:
    - file_path: path to CleanSidamaForPrediction.xlsx
    - output_path: path for output file (optional)
    
    Returns:
    - Filled DataFrame
    """
    print("🌧️  FILLING PRECIPITATION DATA FOR SIDAMA REGION")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} records")
    
    # Define precipitation columns and regional averages
    precip_columns = ['Precipitation Oct-Jan', 'Feb-May', 'Jun-Sep']
    regional_averages = {
        'Precipitation Oct-Jan': 176.7,  # Main rains
        'Feb-May': 145.1,               # Small rains  
        'Jun-Sep': 25.9                 # Dry season
    }
    
    # Clean data - replace fill values with NaN
    print("Cleaning fill values...")
    fill_values = [-9999, -999, 99999, 9999]
    for col in precip_columns:
        if col in df.columns:
            for fill_val in fill_values:
                df[col] = df[col].replace(fill_val, np.nan)
            # Also remove negative values and extreme outliers
            df.loc[(df[col] < 0) | (df[col] > 2000), col] = np.nan
    
    # Identify records with valid precipitation data
    has_valid_precip = df[precip_columns].notna().any(axis=1)
    valid_data = df[has_valid_precip].copy()
    
    print(f"Records with some valid precipitation data: {len(valid_data)}")
    
    # Statistics before filling
    print("\\nBefore filling:")
    for col in precip_columns:
        if col in df.columns:
            valid_count = df[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%) valid")
    
    # Fill missing values
    print("\\nFilling missing precipitation values...")
    filled_count = {'spatial': 0, 'regional': 0, 'failed': 0}
    
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(df)} records...")
        
        # Check if any precipitation values are missing
        missing_cols = [col for col in precip_columns 
                       if col in df.columns and pd.isna(row[col])]
        
        if missing_cols:
            target_lat = row['latitude']
            target_lon = row['longitude']
            
            if pd.notna(target_lat) and pd.notna(target_lon):
                # Attempt spatial interpolation
                interpolated = spatial_interpolation(
                    target_lat, target_lon, valid_data, missing_cols,
                    max_distance_deg=0.1, k=5  # ~10km radius, 5 neighbors
                )
                
                # Fill values
                for col in missing_cols:
                    if interpolated and col in interpolated and interpolated[col] is not None:
                        # Use spatial interpolation
                        df.at[idx, col] = interpolated[col]
                        filled_count['spatial'] += 1
                    elif col in regional_averages:
                        # Use regional average as fallback
                        df.at[idx, col] = regional_averages[col]
                        filled_count['regional'] += 1
                    else:
                        filled_count['failed'] += 1
    
    # Statistics after filling
    print("\\n" + "=" * 60)
    print("FILLING RESULTS")
    print("=" * 60)
    
    print(f"\\nFilled using spatial interpolation: {filled_count['spatial']} values")
    print(f"Filled using regional averages: {filled_count['regional']} values")
    print(f"Failed to fill: {filled_count['failed']} values")
    
    print("\\nAfter filling:")
    for col in precip_columns:
        if col in df.columns:
            valid_count = df[col].notna().sum()
            improvement = valid_count - df[col].notna().sum() if col in df.columns else 0
            print(f"  {col}: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%) valid")
    
    # Quality check - show filled value ranges
    print("\\nQuality check - filled value ranges:")
    for col in precip_columns:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: {values.min():.1f} - {values.max():.1f} mm (mean: {values.mean():.1f})")
    
    # Save results
    if output_path is None:
        output_path = file_path.replace('.xlsx', '_filled.xlsx')
    
    print(f"\\nSaving filled data to: {output_path}")
    df.to_excel(output_path, index=False)
    
    print("\\n✅ Precipitation filling completed successfully!")
    return df

def validate_filled_data(df, precip_columns):
    """
    Validate the quality of filled precipitation data
    """
    print("\\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    # Check for reasonable values
    seasonal_ranges = {
        'Precipitation Oct-Jan': (150, 200),    # Main rains
        'Feb-May': (100, 200),                  # Small rains
        'Jun-Sep': (15, 40)                     # Dry season
    }
    
    for col in precip_columns:
        if col in df.columns:
            values = df[col].dropna()
            min_val, max_val = seasonal_ranges.get(col, (0, 1000))
            
            valid_range = values[(values >= min_val) & (values <= max_val)]
            out_of_range = len(values) - len(valid_range)
            
            print(f"\\n{col}:")
            print(f"  Total values: {len(values)}")
            print(f"  In expected range ({min_val}-{max_val}mm): {len(valid_range)}")
            print(f"  Out of range: {out_of_range}")
            
            if out_of_range > 0:
                print(f"  ⚠️  Warning: {out_of_range} values outside expected range")
            else:
                print(f"  ✅ All values in reasonable range")

# Main execution function
def main():
    """
    Main function to fill precipitation data
    """
    # File paths - update these with your actual paths
    input_file = r"D:\\Work Folder\\maaz Work Africa water level\\OneDrive_2025-01-06\\SGG zones\\CleanSidamaForPrediction.xlsx"
    output_file = r"D:\\Work Folder\\maaz Work Africa water level\\OneDrive_2025-01-06\\SGG zones\\CleanSidamaForPrediction_filled.xlsx"
    
    try:
        # Fill precipitation data
        df_filled = fill_precipitation_data(input_file, output_file)
        
        # Validate results
        precip_columns = ['Precipitation Oct-Jan', 'Feb-May', 'Jun-Sep']
        validate_filled_data(df_filled, precip_columns)
        
        print(f"\\n🎉 SUCCESS! Filled precipitation data saved to:")
        print(f"   {output_file}")
        
        return df_filled
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return None

if __name__ == "__main__":
    filled_data = main()