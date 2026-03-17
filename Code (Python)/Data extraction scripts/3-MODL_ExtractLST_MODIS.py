import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

def extract_data_point(filename, coordinates):
    """Extract data from a single MODIS TIF file at all point locations."""
    if not Path(filename).exists():
        print(f"File not found: {filename}")
        return np.full(len(coordinates), np.nan)
    
    try:
        with rasterio.open(filename) as src:
            # Convert coordinates to (lon, lat) tuples
            coord_pairs = [(coord[0], coord[1]) for coord in coordinates]
            
            # Sample data at point locations
            samples = list(src.sample(coord_pairs, indexes=1))
            
            # Convert to values and apply scale factor
            data = []
            for sample in samples:
                if len(sample) > 0 and sample[0] != src.nodata:
                    # Apply scale factor (0.02 for LST Day 1km)
                    data.append(sample[0] * 0.02)
                else:
                    data.append(np.nan)
            
            return np.array(data)
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return np.full(len(coordinates), np.nan)

def extract_all_data(data_path, point_data_path, years, day_ranges):
    """Extract data for all points and seasonal day ranges."""
    
    # Load point coordinates - handle the Easting/Northing mislabeling
    point_data = pd.read_csv(point_data_path)
    
    # Use Easting/Northing as Longitude/Latitude (they're geographic coordinates)
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
    
    # Initialize empty dictionary to store extracted data
    all_data = {}
    
    for yr in years:
        print(f"Processing year {yr}...")
        all_data[yr] = {}
        
        for i, day_range in enumerate(day_ranges):
            season_name = f"season_{i+1}_days_{min(day_range)}-{max(day_range)}"
            print(f"  Processing {season_name}...")
            
            data_list = []
            files_found = 0
            
            for day in day_range:
                # Updated filename pattern for your TIF files
                filename = f"{data_path}/MOD11A1.061_LST_Day_1km_doy{yr}{day:03d}_aid0001.tif"
                
                if Path(filename).exists():
                    files_found += 1
                    data_point = extract_data_point(filename, point_coordinates)
                    data_list.append(data_point)
                else:
                    # Skip missing files but track them
                    continue
            
            print(f"    Found {files_found}/{len(day_range)} files for this season")
            
            if files_found > 0:
                # Convert to numpy array and transpose (points x days)
                all_data[yr][season_name] = np.array(data_list).T
            else:
                print(f"    No files found for {season_name}")
                all_data[yr][season_name] = np.full((len(point_coordinates), 1), np.nan)
    
    return all_data

def main():
    """Main function following the original logic exactly."""
    
    # Define paths
    data_path = "LST/LST"  # Updated path format
    point_data_path = "GridPoints_Predicted_Sidama.csv"
    output_path = "LST_ExtractedData"  # Updated output path
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Check what years are available in your dataset
    data_dir = Path(data_path)
    available_files = list(data_dir.glob("*.tif"))
    available_years = set()
    available_days_by_year = {}
    
    if available_files:
        print("Analyzing available data...")
        for f in available_files:
            try:
                # Extract year and day from filename: MOD11A1.061_LST_Day_1km_doy2023XXX_aid0001.tif
                filename = f.name
                if 'doy' in filename:
                    year_day = filename.split('doy')[1][:7]  # Get year + day part
                    year = year_day[:4]
                    day = int(year_day[4:])
                    
                    available_years.add(year)
                    if year not in available_days_by_year:
                        available_days_by_year[year] = []
                    available_days_by_year[year].append(day)
            except:
                continue
        
        # Sort and display available data
        available_years = sorted(list(available_years))
        print(f"Available years: {available_years}")
        
        for year in available_years:
            days = sorted(available_days_by_year[year])
            print(f"  {year}: days {min(days)} to {max(days)} ({len(days)} files)")
    
    # Determine years to process
    if len(available_years) >= 2:
        # Use 2 years like original (ideally consecutive years)
        years = available_years[:2]
        print(f"Processing 2 years of data: {years} (like original)")
    elif len(available_years) == 1:
        # Use single year available
        years = available_years
        print(f"Processing 1 year of data: {years}")
    else:
        print("No recognizable data files found!")
        return
    
    # Original day ranges (for 2-year averaging like the original)
    day_ranges = [
        range(234, 244),  # Late summer (days 234-243)
        range(274, 325),  # Fall (days 274-324)
        range(326, 357),  # Early winter (days 326-356) 
        range(358, 366)   # Late winter (days 358-365)
    ]
    
    # Check if original day ranges exist in your data
    all_available_days = []
    for year in years:
        if year in available_days_by_year:
            all_available_days.extend(available_days_by_year[year])
    
    if all_available_days:
        min_day, max_day = min(all_available_days), max(all_available_days)
        print(f"Available day range across all years: {min_day} to {max_day}")
        
        # Check if original ranges are available
        original_ranges_available = all(
            any(day in all_available_days for day in day_range) 
            for day_range in day_ranges
        )
        
        if not original_ranges_available:
            print("Original day ranges not fully available. Using quarterly ranges based on available data:")
            day_ranges = [
                range(min_day, min_day + (max_day - min_day)//4),
                range(min_day + (max_day - min_day)//4, min_day + (max_day - min_day)//2),
                range(min_day + (max_day - min_day)//2, min_day + 3*(max_day - min_day)//4),
                range(min_day + 3*(max_day - min_day)//4, max_day + 1)
            ]
    
    print(f"Using day ranges: {[(min(r), max(r)) for r in day_ranges]}")
    
    # Extract data for all points and days (following original logic exactly)
    all_data = extract_all_data(data_path, point_data_path, years, day_ranges)
    
    # Process and save extracted data (EXACTLY like original - average across years)
    print("\nAveraging across years and seasons...")
    for i, day_range in enumerate(day_ranges):
        season_name = f"season_{i+1}_days_{min(day_range)}-{max(day_range)}"
        
        # Collect data from all years for this season
        all_years_data = []
        for yr in years:
            if yr in all_data and season_name in all_data[yr]:
                year_data = all_data[yr][season_name]
                if year_data.size > 0:
                    # Average across days within this year
                    year_seasonal_mean = np.nanmean(year_data, axis=1)
                    all_years_data.append(year_seasonal_mean)
        
        if all_years_data:
            # Average across years (like original 2-year approach)
            if len(all_years_data) > 1:
                combined_data = np.nanmean(np.array(all_years_data), axis=0)
                print(f"Averaged {len(all_years_data)} years for {season_name}")
            else:
                combined_data = all_years_data[0]
                print(f"Used single year for {season_name}")
            
            # Save data to CSV (following original naming pattern)
            day_range_str = f"{min(day_range)}-{max(day_range)}"
            if len(years) > 1:
                year_str = f"{years[0]}-{years[-1]}"
            else:
                year_str = years[0]
            
            output_filename = f"{output_path}/LSTDay_GridPoints_{year_str}_Days{day_range_str}.csv"
            
            # Save as single column like original
            np.savetxt(output_filename, combined_data, delimiter=",")
            
            print(f"Saved: {output_filename}")
            print(f"  Shape: {combined_data.shape}")
            print(f"  Temperature range: {np.nanmin(combined_data):.1f}°C to {np.nanmax(combined_data):.1f}°C")
        else:
            print(f"No data available for {season_name}")
    
    print("Data extraction completed!")

if __name__ == "__main__":
    main()