import os
import glob
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask
from shapely.geometry import Point
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPMDataDiscovery:
    """Module for discovering and organizing GPM HDF5 files."""
    
    def __init__(self, data_paths: List[str]):
        self.data_paths = data_paths
        self.file_inventory = []
    
    def scan_directories(self) -> pd.DataFrame:
        """Scan directories for HDF5 files and extract metadata."""
        logger.info("Scanning directories for GPM HDF5 files...")
        
        for path in self.data_paths:
            if os.path.exists(path):
                # Search for HDF5 files recursively
                pattern = os.path.join(path, "**/*.HDF5")
                files = glob.glob(pattern, recursive=True)
                
                for file_path in files:
                    file_info = self._extract_file_metadata(file_path)
                    if file_info:
                        self.file_inventory.append(file_info)
            else:
                logger.warning(f"Path does not exist: {path}")
        
        df = pd.DataFrame(self.file_inventory)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        logger.info(f"Found {len(df)} GPM files")
        return df
    
    def _extract_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Extract date and metadata from filename."""
        filename = os.path.basename(file_path)
        
        # Pattern for GPM files: 3B-MO.MS.MRG.3IMERG.YYYYMMDD-SXXXXXX-EXXXXXX.XX.V07B.HDF5
        pattern = r'3B-MO\.MS\.MRG\.3IMERG\.(\d{8})-S\d{6}-E\d{6}\.\d{2}\.V\d{2}[AB]\.HDF5'
        match = re.search(pattern, filename)
        
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day
                }
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")
        else:
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
        
        return None
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter files by date range."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        filtered = df[(df['date'] >= start) & (df['date'] <= end)]
        logger.info(f"Filtered to {len(filtered)} files between {start_date} and {end_date}")
        return filtered

class GPMFileProcessor:
    """Module for processing individual GPM HDF5 files."""
    
    def __init__(self):
        self.temp_raster_path = "temp_precipitation_raster.tif"
    
    def read_gpm_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Read GPM HDF5 file and extract coordinates and precipitation data."""
        try:
            with Dataset(file_path, 'r') as nc_data:
                lon = nc_data.variables['Grid/lon'][:]
                lat = nc_data.variables['Grid/lat'][:]
                prec_array = nc_data.variables['Grid/precipitation'][:]
                fill_value = nc_data.variables['Grid/precipitation']._FillValue
                
                # Replace fill values with NaN
                prec_array = np.where(prec_array == fill_value, np.nan, prec_array)
                
                return lon, lat, prec_array, fill_value
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def create_temporary_raster(self, lon: np.ndarray, lat: np.ndarray, 
                              prec_array: np.ndarray) -> str:
        """Create a temporary GeoTIFF raster from precipitation data."""
        transform = from_bounds(np.min(lon), np.min(lat), np.max(lon), np.max(lat), 
                              len(lon), len(lat))
        
        with rasterio.open(
            self.temp_raster_path,
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
        
        return self.temp_raster_path
    
    def cleanup_temp_files(self):
        """Remove temporary raster files."""
        if os.path.exists(self.temp_raster_path):
            os.remove(self.temp_raster_path)

class CoordinateExtractor:
    """Module for extracting precipitation values at specific coordinates."""
    
    def __init__(self, coordinates_file: str = None, coordinates_df: pd.DataFrame = None):
        if coordinates_file:
            self.coordinates_df = pd.read_csv(coordinates_file)
        elif coordinates_df is not None:
            self.coordinates_df = coordinates_df
        else:
            raise ValueError("Either coordinates_file or coordinates_df must be provided")
        
        self._validate_coordinates()
    
    def _validate_coordinates(self):
        """Validate that coordinate columns exist."""
        # Try to identify longitude and latitude columns
        possible_lon_cols = ['longitude', 'Longitude', 'lon', 'Lon', 'long', 'Long']
        possible_lat_cols = ['latitude', 'Latitude', 'lat', 'Lat']
        
        self.lon_col = None
        self.lat_col = None
        
        for col in possible_lon_cols:
            if col in self.coordinates_df.columns:
                self.lon_col = col
                break
        
        for col in possible_lat_cols:
            if col in self.coordinates_df.columns:
                self.lat_col = col
                break
        
        if not self.lon_col or not self.lat_col:
            available_cols = list(self.coordinates_df.columns)
            raise ValueError(f"Could not find longitude/latitude columns. Available columns: {available_cols}")
        
        logger.info(f"Using columns: {self.lon_col} (longitude), {self.lat_col} (latitude)")
    
    def extract_values_from_raster(self, raster_path: str) -> List[float]:
        """Extract precipitation values for all coordinates from raster."""
        values = []
        
        with rasterio.open(raster_path) as src:
            for _, row in self.coordinates_df.iterrows():
                lon, lat = row[self.lon_col], row[self.lat_col]
                point = Point(lon, lat)
                
                try:
                    # Use mask to extract value at point
                    masked_data, _ = mask(src, [point], crop=True, all_touched=True)
                    masked_data = masked_data.flatten()
                    
                    # Get first non-NaN value, or NaN if all are NaN
                    non_nan_values = masked_data[~np.isnan(masked_data)]
                    if len(non_nan_values) > 0:
                        values.append(non_nan_values[0])
                    else:
                        values.append(np.nan)
                        
                except Exception as e:
                    logger.warning(f"Error extracting value at ({lon}, {lat}): {e}")
                    values.append(np.nan)
        
        return values

class SeasonalAggregator:
    """Module for calculating seasonal precipitation averages."""
    
    def __init__(self):
        self.season_definitions = {
            'Oct-Jan': [10, 11, 12, 1],      # Dry season
            'Feb-May': [2, 3, 4, 5],         # Small rains
            'Jun-Sep': [6, 7, 8, 9]          # Big rains
        }
        
        # Days in each month
        self.days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
    
    def convert_to_cumulative_mm(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """Convert mm/hour to cumulative mm by multiplying by hours and days."""
        cumulative_data = monthly_data.copy()
        
        for col in monthly_data.columns:
            if isinstance(col, str) and len(col) >= 6:
                try:
                    # Extract month from column name (assuming format like YYYYMM)
                    month = int(col[-2:])
                    days = self.days_in_month[month]
                    
                    # Convert mm/hour to mm/month (multiply by 24 hours * days in month)
                    cumulative_data[col] = monthly_data[col] * 24 * days
                    
                except (ValueError, KeyError):
                    logger.warning(f"Could not determine month for column: {col}")
        
        return cumulative_data
    
    def calculate_seasonal_averages(self, monthly_data: pd.DataFrame, 
                                  target_years: List[int]) -> pd.DataFrame:
        """Calculate seasonal averages for specified years."""
        # Convert to cumulative precipitation
        cumulative_data = self.convert_to_cumulative_mm(monthly_data)
        
        seasonal_results = {}
        
        for season_name, months in self.season_definitions.items():
            seasonal_values = []
            
            for year in target_years:
                year_values = []
                
                for month in months:
                    # Handle year transition for Oct-Jan season
                    if season_name == 'Oct-Jan' and month <= 1:
                        col_name = f"{year+1}{month:02d}"
                    else:
                        col_name = f"{year}{month:02d}"
                    
                    if col_name in cumulative_data.columns:
                        year_values.append(cumulative_data[col_name])
                
                if year_values:
                    # Calculate mean for this year
                    year_mean = pd.concat(year_values, axis=1).mean(axis=1)
                    seasonal_values.append(year_mean)
            
            if seasonal_values:
                # Calculate overall seasonal average across years
                seasonal_results[season_name] = pd.concat(seasonal_values, axis=1).mean(axis=1)
        
        return pd.DataFrame(seasonal_results)

class GPMPrecipitationExtractor:
    """Main orchestrator class for GPM precipitation extraction."""
    
    def __init__(self, data_paths: List[str], coordinates_source: str = None):
        self.data_paths = data_paths
        self.coordinates_source = coordinates_source
        
        # Initialize modules
        self.discovery = GPMDataDiscovery(data_paths)
        self.processor = GPMFileProcessor()
        self.aggregator = SeasonalAggregator()
        
        # Data storage
        self.file_inventory = None
        self.monthly_precipitation = None
        self.seasonal_precipitation = None
    
    def discover_files(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Discover all GPM files in the specified date range."""
        self.file_inventory = self.discovery.scan_directories()
        
        if start_date and end_date:
            self.file_inventory = self.discovery.filter_by_date_range(
                self.file_inventory, start_date, end_date
            )
        
        return self.file_inventory
    
    def extract_monthly_precipitation(self, coordinate_extractor: CoordinateExtractor) -> pd.DataFrame:
        """Extract monthly precipitation data for all coordinates."""
        if self.file_inventory is None:
            raise ValueError("Must run discover_files() first")
        
        monthly_data = {}
        
        for _, file_info in self.file_inventory.iterrows():
            file_path = file_info['file_path']
            date_key = f"{file_info['year']}{file_info['month']:02d}"
            
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            
            try:
                # Read GPM file
                lon, lat, prec_array, fill_value = self.processor.read_gpm_file(file_path)
                
                # Create temporary raster
                raster_path = self.processor.create_temporary_raster(lon, lat, prec_array)
                
                # Extract values at coordinates
                values = coordinate_extractor.extract_values_from_raster(raster_path)
                monthly_data[date_key] = values
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
            finally:
                # Cleanup temporary files
                self.processor.cleanup_temp_files()
        
        self.monthly_precipitation = pd.DataFrame(monthly_data)
        return self.monthly_precipitation
    
    def calculate_seasonal_averages(self, target_years: List[int]) -> pd.DataFrame:
        """Calculate seasonal precipitation averages."""
        if self.monthly_precipitation is None:
            raise ValueError("Must run extract_monthly_precipitation() first")
        
        self.seasonal_precipitation = self.aggregator.calculate_seasonal_averages(
            self.monthly_precipitation, target_years
        )
        
        return self.seasonal_precipitation
    
    def save_results(self, output_dir: str = ".", prefix: str = "GPM_precipitation"):
        """Save results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.monthly_precipitation is not None:
            monthly_path = os.path.join(output_dir, f"{prefix}_monthly.csv")
            self.monthly_precipitation.to_csv(monthly_path, index=False)
            logger.info(f"Saved monthly precipitation to: {monthly_path}")
        
        if self.seasonal_precipitation is not None:
            seasonal_path = os.path.join(output_dir, f"{prefix}_seasonal.csv")
            self.seasonal_precipitation.to_csv(seasonal_path, index=False)
            logger.info(f"Saved seasonal precipitation to: {seasonal_path}")
    
    def run_full_extraction(self, coordinates_source: str, target_years: List[int],
                          start_date: str = None, end_date: str = None,
                          output_dir: str = ".", prefix: str = "GPM_precipitation") -> pd.DataFrame:
        """Run the complete extraction workflow."""
        
        # Step 1: Discover files
        logger.info("Step 1: Discovering GPM files...")
        self.discover_files(start_date, end_date)
        
        # Step 2: Initialize coordinate extractor
        logger.info("Step 2: Loading coordinates...")
        coordinate_extractor = CoordinateExtractor(coordinates_file=coordinates_source)
        
        # Step 3: Extract monthly precipitation
        logger.info("Step 3: Extracting monthly precipitation...")
        self.extract_monthly_precipitation(coordinate_extractor)
        
        # Step 4: Calculate seasonal averages
        logger.info("Step 4: Calculating seasonal averages...")
        seasonal_results = self.calculate_seasonal_averages(target_years)
        
        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        self.save_results(output_dir, prefix)
        
        logger.info("Extraction complete!")
        return seasonal_results

# Example usage
if __name__ == "__main__":
    # Configuration
    data_paths = [
        r"D:\Work Folder\maaz Work Africa water level\GPM_3IMERGM_data_apr282024to2025\GPM_3IMERGM_07-20250427_224012",
        r"D:\Work Folder\maaz Work Africa water level\GPM_3IMERGM_data_apr282024to2025\GPM_3IMERGM_07-20250427_223225"
    ]
    
    coordinates_file = "your_coordinates_file.csv"  # Update with your coordinates file
    target_years = [2022, 2023]  # Specify which years to use for 2-year average
    
    # Initialize extractor
    extractor = GPMPrecipitationExtractor(data_paths)
    
    # Run full extraction
    seasonal_results = extractor.run_full_extraction(
        coordinates_source=coordinates_file,
        target_years=target_years,
        start_date="2022-01-01",
        end_date="2023-12-31",
        output_dir="precipitation_results",
        prefix="bilate_precipitation"
    )
    
    print("\nSeasonal precipitation results:")
    print(seasonal_results.head())
    print(f"\nResults shape: {seasonal_results.shape}")
    print(f"Seasons calculated: {list(seasonal_results.columns)}")
