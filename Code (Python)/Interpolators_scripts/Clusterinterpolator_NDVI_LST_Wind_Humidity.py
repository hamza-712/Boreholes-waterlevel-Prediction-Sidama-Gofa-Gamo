import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def fast_hybrid_interpolation(file_path, output_path=None):
    """
    FAST version using vectorized operations and KDTree for spatial search
    FIXED: Now processes ALL records, not just the subset with existing climate data
    """
    print("🚀 FAST HYBRID SPATIAL INTERPOLATION - OPTION C (FIXED)")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} records")
    
    # Define variable groups with CORRECT coordinate mappings
    variable_groups = {
        'LST': {
            'columns': ['LST-234-243', 'LST-273-324', 'LST-326-356', 'LST-358-365'],
            'max_distance_km': 10,
            'max_distance_deg': 0.09,
            'coord_columns': ['longitude', 'latitude']  # Environmental data coordinates
        },
        'NDVI': {
            'columns': ['NDVI-June-Sep', 'NDVI-dry-Oct-Jan', 'Feb-May.1'],
            'max_distance_km': 10,
            'max_distance_deg': 0.09,
            'coord_columns': ['longitude', 'latitude']  # Environmental data coordinates
        },
        'Wind_Speed': {
            'columns': ['WindSpeedMeanJunToSep', 'WindSpeedMeanOctToJan', 'WindSpeedMeanFebToMay'],
            'max_distance_km': 5,
            'max_distance_deg': 0.045,
            'coord_columns': ['longitude', 'latitude']  # Environmental data coordinates
        },
        'Specific_Humidity': {
            'columns': ['SpecificHumidity_meanCumJunToSep', 'SpecificHumidity_meanCumOctToJan', 'SpecificHumidity_meanCumFebToMay'],
            'max_distance_km': 5,
            'max_distance_deg': 0.045,
            'coord_columns': ['longitude', 'latitude']  # Environmental data coordinates
        }
    }
    
    # Clean data quickly
    print("Cleaning data...")
    all_columns = []
    for group in variable_groups.values():
        all_columns.extend(group['columns'])
    
    # Vectorized cleaning
    for col in all_columns:
        if col in df.columns:
            # Replace fill values
            df[col] = df[col].replace([-9999, -999, 99999, 9999], np.nan)
            
            # Apply range filters
            if 'LST' in col:
                df.loc[(df[col] < -50) | (df[col] > 80), col] = np.nan
            elif 'NDVI' in col:
                df.loc[(df[col] < -1) | (df[col] > 1), col] = np.nan
            elif 'Wind' in col:
                df.loc[(df[col] < 0) | (df[col] > 50), col] = np.nan
            elif 'Humidity' in col:
                df.loc[(df[col] < 0) | (df[col] > 0.1), col] = np.nan
    
    # Calculate regional averages quickly
    print("Calculating regional averages...")
    regional_averages = {}
    for group_name, group_info in variable_groups.items():
        for col in group_info['columns']:
            if col in df.columns:
                regional_avg = df[col].mean()
                regional_averages[col] = regional_avg
                print(f"  {col}: {regional_avg:.4f}")
    
    print("\\nBefore interpolation:")
    for col in all_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing}/{len(df)} ({missing/len(df)*100:.1f}%) missing")
    
    # Process each variable group
    interpolation_stats = {'spatial': 0, 'regional': 0, 'total_processed': 0}
    
    for group_name, group_info in variable_groups.items():
        print(f"\\n📊 Processing {group_name} (fast method)...")
        
        for col in group_info['columns']:
            if col not in df.columns:
                continue
                
            print(f"  {col}...")
            
            # FIXED: Use correct coordinate columns for this variable group
            lon_col, lat_col = group_info['coord_columns']
            
            # Get valid and missing data using CORRECT coordinates
            valid_mask = (df[col].notna() & 
                         df[lat_col].notna() & 
                         df[lon_col].notna())
            missing_mask = (df[col].isna() & 
                           df[lat_col].notna() & 
                           df[lon_col].notna())
            
            valid_data = df[valid_mask]
            missing_indices = df[missing_mask].index
            
            if len(valid_data) == 0 or len(missing_indices) == 0:
                print(f"    No interpolation needed (valid: {len(valid_data)}, missing: {len(missing_indices)})")
                continue
            
            print(f"    Valid: {len(valid_data)}, Missing: {len(missing_indices)}")
            print(f"    Using coordinates: {lon_col}, {lat_col}")
            
            # Build KDTree for fast spatial search using CORRECT coordinates
            valid_coords = valid_data[[lat_col, lon_col]].values
            tree = cKDTree(valid_coords)
            
            # Get coordinates of missing points using CORRECT coordinates
            missing_coords = df.loc[missing_indices, [lat_col, lon_col]].values
            
            # Find neighbors for all missing points at once
            distances, indices = tree.query(
                missing_coords, 
                k=min(5, len(valid_data)), 
                distance_upper_bound=group_info['max_distance_deg']
            )
            
            # Process results
            spatial_fills = 0
            regional_fills = 0
            
            # Process in batches for progress tracking
            batch_size = 5000
            total_batches = (len(missing_indices) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(missing_indices))
                
                print(f"    Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
                
                for i in range(start_idx, end_idx):
                    idx = missing_indices[i]
                    
                    # Get neighbors for this point
                    point_distances = distances[i]
                    point_indices = indices[i]
                    
                    # Filter valid neighbors (not inf distance)
                    valid_neighbors = point_distances != np.inf
                    
                    if np.sum(valid_neighbors) >= 2:
                        # Spatial interpolation
                        neighbor_distances = point_distances[valid_neighbors]
                        neighbor_data_indices = valid_data.iloc[point_indices[valid_neighbors]].index
                        neighbor_values = df.loc[neighbor_data_indices, col].values
                        
                        # Remove any NaN values
                        non_nan_mask = ~np.isnan(neighbor_values)
                        if np.sum(non_nan_mask) >= 2:
                            clean_distances = neighbor_distances[non_nan_mask]
                            clean_values = neighbor_values[non_nan_mask]
                            
                            # Inverse distance weighting
                            weights = 1 / np.maximum(clean_distances, 1e-6) ** 2
                            interpolated_value = np.sum(clean_values * weights) / np.sum(weights)
                            
                            df.at[idx, col] = interpolated_value
                            spatial_fills += 1
                            continue
                    
                    # Fallback to regional average
                    if pd.notna(regional_averages.get(col)):
                        df.at[idx, col] = regional_averages[col]
                        regional_fills += 1
            
            print(f"    Filled: {spatial_fills:,} spatial + {regional_fills:,} regional")
            interpolation_stats['spatial'] += spatial_fills
            interpolation_stats['regional'] += regional_fills
            interpolation_stats['total_processed'] += len(missing_indices)
    
    # Final results
    print("\\n" + "=" * 60)
    print("🎯 FAST INTERPOLATION RESULTS")
    print("=" * 60)
    
    print(f"\\nFills by method:")
    print(f"  Spatial interpolation: {interpolation_stats['spatial']:,}")
    print(f"  Regional averages: {interpolation_stats['regional']:,}")
    print(f"  Total processed: {interpolation_stats['total_processed']:,}")
    
    print("\\nAfter interpolation:")
    for col in all_columns:
        if col in df.columns:
            complete_pct = (df[col].notna().sum() / len(df)) * 100
            missing_count = df[col].isna().sum()
            print(f"  {col}: {complete_pct:.1f}% complete ({missing_count:,} still missing)")
    
    # Count complete records for key variables using CORRECT coordinates
    print("\\nCoordinate usage summary:")
    env_coords = df[['longitude', 'latitude']].dropna()
    elev_coords = df[['longitude.1', 'latitude.1']].dropna() if 'longitude.1' in df.columns else pd.DataFrame()
    
    print(f"  Environmental coordinates (lon, lat): {len(env_coords):,} records")
    if len(elev_coords) > 0:
        print(f"  Elevation coordinates (lon.1, lat.1): {len(elev_coords):,} records")
    
    # Calculate complete records based on environmental coordinates (since that's what we use for climate data)
    key_vars = [col for col in all_columns if col in df.columns]
    complete_records = df[df['longitude'].notna() & df['latitude'].notna()][key_vars].dropna().shape[0]
    total_env_records = len(env_coords)
    
    print(f"\\nComplete records (all climate variables): {complete_records:,}/{total_env_records:,} ({complete_records/total_env_records*100:.1f}%)")
    
    # Save results
    if output_path is None:
        output_path = file_path.replace('.xlsx', '_fast_interpolated.xlsx')
    
    print(f"\\nSaving to: {output_path}")
    df.to_excel(output_path, index=False)
    
    print("\\n🎉 Fast interpolation completed!")
    return df

def main():
    """Main function for fast interpolation"""
    input_file = r"D:\\Work Folder\\maaz Work Africa water level\\OneDrive_2025-01-06\\SGG zones\\CleanSidamaForPrediction_filled.xlsx"
    output_file = r"D:\\Work Folder\\maaz Work Africa water level\\OneDrive_2025-01-06\\SGG zones\\CleanSidamaForPrediction_FAST_interpolated.xlsx"
    
    try:
        result = fast_hybrid_interpolation(input_file, output_file)
        print(f"\\n✅ SUCCESS! Use this file in your dataset merger:")
        print(f"   {output_file}")
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    fast_result = main()