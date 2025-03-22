# """Module for handling forest cover and forest change data processing."""

# import os
# import logging
# import sys
# from pathlib import Path
# from typing import Dict, Any, List, Optional
# from tqdm import tqdm
# import ee
# from google.cloud import bigquery

# # Add parent directory to path for direct script execution
# if __name__ == "__main__":
#     sys.path.append(str(Path(__file__).parent.parent))
#     from handlers.base_handler import BaseHandler
# else:
#     from .base_handler import BaseHandler


# class ForestHandler(BaseHandler):
#     """Handles forest data processing and analysis using Google Earth Engine."""

#     def __init__(self):
#         """Initialize the ForestHandler."""
#         super().__init__()
#         self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

#         # Initialize Earth Engine
#         try:
#             ee.Initialize()
#         except ee.EEException as e:
#             self.logger.error("Failed to initialize Earth Engine: %s", str(e))
#             raise

#     def get_forest_data(self, bounds: Dict[str, float]) -> Dict[str, Any]:
#         """
#         Fetch forest cover and total loss data for a bounding box.
        
#         Args:
#             bounds (dict): Dictionary with keys: min_lon, max_lon, min_lat, max_lat
            
#         Returns:
#             Dictionary containing forest data for the grid cell
            
#         Raises:
#             ee.EEException: If Earth Engine operations fail
#             ValueError: If input parameters are invalid
#         """
#         try:
#             # Load Hansen dataset (2023 version)
#             hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

#             # Create the region of interest
#             roi = ee.Geometry.Rectangle([
#                 bounds['min_lon'],
#                 bounds['min_lat'],
#                 bounds['max_lon'],
#                 bounds['max_lat']
#             ])

#             # Get tree cover from 2000
#             treecover2000 = hansen.select(['treecover2000'])

#             # Get forest loss year (0-23) and create a mask for any loss
#             lossyear = hansen.select(['lossyear'])
#             total_loss = lossyear.gt(0)  # Mask of all pixels that experienced loss

#             # Get forest gain (2000-2012)
#             gain = hansen.select(['gain'])

#             # Calculate statistics for each grid cell
#             grid_stats = treecover2000.addBands(total_loss).addBands(gain).reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=roi,
#                 scale=250,  # meters per pixel
#                 maxPixels=1e9
#             )

#             # Get the results
#             results = grid_stats.getInfo()

#             return {
#                 'grid_lon': bounds['min_lon'] + (bounds['max_lon'] - bounds['min_lon']) / 2,
#                 'grid_lat': bounds['min_lat'] + (bounds['max_lat'] - bounds['min_lat']) / 2,
#                 'tree_cover_2000': results.get('treecover2000', 0),
#                 'total_forest_loss': results.get('lossyear_gt_0', 0),
#                 'forest_gain': results.get('gain', 0)
#             }

#         except ee.EEException as e:
#             self.logger.error("Earth Engine error: %s", str(e))
#             raise
#         except KeyError as e:
#             self.logger.error("Missing required bound parameter: %s", str(e))
#             raise ValueError(f"Missing required bound parameter: {e}") from e

#     def get_forest_data_batch(self, grid_cells, batch_size=100) -> List[Dict[str, Any]]:
#         """
#         Fetch forest cover and forest loss data for multiple grid cells in a single EE call.
#         Returns data with a year dimension for forest loss.
        
#         Args:
#             grid_cells (List[Dict]): List of grid cells with grid_lon and grid_lat
#             batch_size (int): Size of batches to process at once
            
#         Returns:
#             List of dictionaries containing forest data for each grid cell and year
            
#         Raises:
#             ee.EEException: If Earth Engine operations fail
#             ValueError: If input parameters are invalid
#         """
#         try:
#             # Load Hansen dataset (2023 version)
#             hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
#             results = []
#             for i in range(0, len(grid_cells), batch_size):
#                 batch = grid_cells[i:i + batch_size]

#                 # Create feature collection of grid cells
#                 features = []
#                 for cell in batch:
#                     grid_lon = float(cell.grid_lon)
#                     grid_lat = float(cell.grid_lat)
#                     bounds = {
#                         'min_lon': grid_lon - 0.125,
#                         'max_lon': grid_lon + 0.125,
#                         'min_lat': grid_lat - 0.125,
#                         'max_lat': grid_lat + 0.125
#                     }

#                     geometry = ee.Geometry.Rectangle([
#                         bounds['min_lon'],
#                         bounds['min_lat'],
#                         bounds['max_lon'],
#                         bounds['max_lat']
#                     ])

#                     feature = ee.Feature(geometry, {
#                         'grid_lon': grid_lon,
#                         'grid_lat': grid_lat,
#                         'grid_lon_index': int((grid_lon + 180) * 4),
#                         'grid_lat_index': int((grid_lat + 90) * 4)
#                     })
#                     features.append(feature)
#                 # Create feature collection
#                 fc = ee.FeatureCollection(features)

#                 # Get tree cover from 2000
#                 treecover2000 = hansen.select(['treecover2000'])

#                 # Get forest loss year and create masks for each year
#                 lossyear = hansen.select(['lossyear'])

#                 # Create an image with bands for each year's loss
#                 loss_by_year = ee.Image.cat([
#                     lossyear.eq(year).rename(f'loss_{year}')
#                     for year in range(1, 24)  # 2001-2023
#                 ])

#                 # Get forest gain (2000-2012)
#                 gain = hansen.select(['gain'])

#                 # Combine all bands
#                 combined = treecover2000.addBands(loss_by_year).addBands(gain)

#                 # Calculate statistics for each grid cell
#                 stats = combined.reduceRegions(
#                     collection=fc,
#                     reducer=ee.Reducer.mean(),
#                     scale=250  # meters per pixel
#                 )

#                 # Get the results
#                 batch_results = stats.getInfo()['features']

#                 # Process results
#                 for feature in batch_results:
#                     props = feature['properties']
#                     grid_lon = props['grid_lon']
#                     grid_lat = props['grid_lat']
#                     grid_lon_index = props['grid_lon_index']
#                     grid_lat_index = props['grid_lat_index']
#                     tree_cover_2000 = props.get('treecover2000', 0)
#                     forest_gain = props.get('gain', 0)

#                     # Add base record for year 2000 (initial state)
#                     results.append({
#                         'grid_lon': grid_lon,
#                         'grid_lat': grid_lat,
#                         'grid_lon_index': grid_lon_index,
#                         'grid_lat_index': grid_lat_index,
#                         'year': 2000,
#                         'tree_cover': tree_cover_2000,
#                         'forest_loss': 0.0,  # No loss in initial year
#                         'forest_gain': forest_gain  # Gain is constant (2000-2012 total)
#                     })

#                     # Add records for each year with loss data (2001-2023)
#                     for year in range(1, 24):
#                         results.append({
#                             'grid_lon': grid_lon,
#                             'grid_lat': grid_lat,
#                             'grid_lon_index': grid_lon_index,
#                             'grid_lat_index': grid_lat_index,
#                             'year': 2000 + year,
#                             'tree_cover': tree_cover_2000,  # Initial tree cover
#                             'forest_loss': props.get(f'loss_{year}', 0),
#                             'forest_gain': forest_gain  # Gain is constant (2000-2012 total)
#                         })

#             return results

#         except ee.EEException as e:
#             self.logger.error("Earth Engine error: %s", str(e))
#             raise
#         except Exception as e:
#             self.logger.error("Error processing batch: %s", str(e))
#             raise

#     def update_forest_data(self):
#         """
#         Update the forest_data table in BigQuery with the latest Hansen data.
#         Process data in grid cells matching our species occurrence data in Africa.
#         Only processes cells that don't already exist in the database.
#         Uses batch processing for improved performance.
        
#         Raises:
#             bigquery.exceptions.BigQueryError: If BigQuery operations fail
#             ee.EEException: If Earth Engine operations fail
#             OSError: If system/IO operations fail
#         """
#         try:
#             client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

#             # Create forest_data table if it doesn't exist
#             self.logger.info("Creating/updating forest_data table...")
#             create_table_query = """
#             CREATE TABLE IF NOT EXISTS `{project_id}.biodiversity.forest_data` (
#                 grid_lon_index INT64,
#                 grid_lat_index INT64,
#                 grid_lon FLOAT64,
#                 grid_lat FLOAT64,
#                 year INT64,
#                 tree_cover FLOAT64,
#                 forest_loss FLOAT64,
#                 forest_gain FLOAT64
#             )
#             CLUSTER BY grid_lon_index, grid_lat_index, year
#             """

#             create_table_query = self.build_query(create_table_query)
#             client.query(create_table_query).result()

#             # Get unique grid cells from species occurrences in Africa
#             self.logger.info("Fetching unique grid cells for Africa...")
#             grid_query = """
#             WITH species_grid_cells AS (
#                 SELECT DISTINCT
#                     ROUND(decimallongitude * 4) / 4 as grid_lon,
#                     ROUND(decimallatitude * 4) / 4 as grid_lat
#                 FROM `{project_id}.biodiversity.occurances_endangered_species_mammals`
#                 WHERE decimallongitude IS NOT NULL
#                     AND decimallatitude IS NOT NULL
#                     -- Filter for Africa's bounds
#                     AND decimallatitude BETWEEN -35 AND 37  -- Roughly from South Africa to Tunisia
#                     AND decimallongitude BETWEEN -17 AND 51  -- Roughly from Western Sahara to Somalia
#             )
#             SELECT s.grid_lon, s.grid_lat
#             FROM species_grid_cells s
#             LEFT JOIN (
#                 SELECT DISTINCT grid_lon, grid_lat 
#                 FROM `{project_id}.biodiversity.forest_data`
#             ) f
#                 ON ROUND(f.grid_lon, 6) = ROUND(s.grid_lon, 6)
#                 AND ROUND(f.grid_lat, 6) = ROUND(s.grid_lat, 6)
#             WHERE f.grid_lon IS NULL
#             """

#             grid_query = self.build_query(grid_query)
#             grid_cells = list(client.query(grid_query).result())
#             total_cells = len(grid_cells)

#             if total_cells == 0:
#                 self.logger.info("No new grid cells to process. Exiting.")
#                 return

#             # Process grid cells in batches
#             batch_size = 1000
#             forest_data = []

#             with tqdm(total=total_cells, desc="Processing grid cells") as pbar:
#                 for i in range(0, total_cells, batch_size):
#                     batch = grid_cells[i:min(i + batch_size, total_cells)]
#                     try:
#                         # Get forest data for this batch
#                         batch_results = self.get_forest_data_batch(batch)
#                         forest_data.extend(batch_results)
#                         # Update progress bar
#                         pbar.update(len(batch))
#                         # Insert batch results
#                         if len(forest_data) >= 1000:
#                             self._insert_forest_data(forest_data)
#                             forest_data = []
#                     except (ee.EEException, ValueError, AttributeError) as e:
#                         self.logger.error(
#                             "Error processing batch starting at index %d: %s", i, str(e))
#                         continue

#             # Insert any remaining records
#             if forest_data:
#                 self._insert_forest_data(forest_data)

#             self.logger.info("Forest data update completed successfully")

#         except bigquery.exceptions.BigQueryError as e:
#             self.logger.error("BigQuery error: %s", str(e))
#             raise
#         except ee.EEException as e:
#             self.logger.error("Earth Engine error: %s", str(e))
#             raise
#         except OSError as e:
#             self.logger.error("System/IO error: %s", str(e))
#             raise

#     def _insert_forest_data(self, forest_data: List[Dict[str, Any]]):
#         """
#         Insert forest data into BigQuery table.
        
#         Args:
#             forest_data (List[Dict]): List of dictionaries containing forest data
            
#         Raises:
#             bigquery.exceptions.BigQueryError: If BigQuery operations fail
#         """
#         try:
#             client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

#             # Create the table reference
#             table_ref = client.dataset('biodiversity').table('forest_data')

#             # Insert the data
#             errors = client.insert_rows_json(table_ref, forest_data)

#             if errors:
#                 raise bigquery.exceptions.BigQueryError(f"Error inserting rows: {errors}")

#         except bigquery.exceptions.BigQueryError as e:
#             self.logger.error("BigQuery error: %s", str(e))
#             raise

#     def calculate_species_forest_correlation(
#         self,
#         species_name: Optional[str] = None,
#         group_by: str = None,
#         country_code: Optional[str] = None,
#         conservation_status: Optional[str] = None,
#         min_observations: int = 10,
#         year: Optional[int] = None
#     ) -> Dict[str, Any]:
#         """Calculate correlation between species observations and forest metrics.
        
#         This method can analyze either a single species or groups of species.
        
#         Args:
#             species_name (str, optional): Scientific name of the species. If None, analyzes groups.
#             group_by (str, optional): How to group species - either 'conservation_status'
#             or 'country' country_code (str, optional): Filter species by country
#             conservation_status (str, optional): Filter species by conservation status
#             min_observations (int): Minimum number of observations required (default: 10)
#             year (int, optional): Specific year to analyze. If None, analyzes aggregated data.
            
#         Returns:
#             dict: Dictionary containing correlation results and statistics
            
#         Raises:
#             ValueError: If invalid parameters are provided
#         """
#         try:
#             # Log and validate parameters
#             self.logger.info("Parameters received: species_name=%s, group_by=%s, country_code=%s, conservation_status=%s, min_observations=%s, year=%s",
#                             species_name, group_by, country_code, conservation_status, min_observations, year)

#             # Handle case where species_name is actually a country_code query
#             if isinstance(species_name, dict) and 'country_code' in species_name:
#                 country_code = species_name['country_code']
#                 species_name = None
#                 group_by = 'country'
#             # Handle case where species_name is a dictionary containing species_name
#             elif isinstance(species_name, dict):
#                 species_name = species_name.get('species_name')

#             client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

#             # Base query structure for aggregating forest metrics
#             forest_metrics = f"""
#                 WITH yearly_forest_metrics AS (
#                     SELECT 
#                         grid_lon,
#                         grid_lat,
#                         year,
#                         tree_cover as initial_tree_cover,
#                         -- Calculate cumulative forest loss up to each year
#                         SUM(CASE 
#                             WHEN tree_cover >= 30 AND forest_loss > 0 
#                             THEN forest_loss 
#                             ELSE 0 
#                         END) OVER (
#                             PARTITION BY grid_lon_index, grid_lat_index
#                             ORDER BY year
#                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
#                         ) as cumulative_loss,
#                         -- Calculate remaining forest cover for each year
#                         tree_cover - SUM(CASE 
#                             WHEN tree_cover >= 30 AND forest_loss > 0 
#                             THEN forest_loss 
#                             ELSE 0 
#                         END) OVER (
#                             PARTITION BY grid_lon_index, grid_lat_index
#                             ORDER BY year
#                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
#                         ) as remaining_tree_cover
#                     FROM `{os.getenv('GOOGLE_CLOUD_PROJECT')}.biodiversity.forest_data`
#                     WHERE tree_cover >= 30  -- Only consider forest areas
#                 )
#                 SELECT 
#                     grid_lon,
#                     grid_lat,
#                     year,
#                     AVG(initial_tree_cover) as initial_tree_cover,
#                     AVG(cumulative_loss) as cumulative_loss,
#                     AVG(remaining_tree_cover) as remaining_tree_cover
#                 FROM yearly_forest_metrics
#                 GROUP BY grid_lon, grid_lat, year
#             """

#             # Handle single species analysis
#             if species_name is not None:
#                 # Single species query and processing
#                 query = f"""
#                 WITH species_counts AS (
#                     SELECT 
#                         ROUND(decimallongitude * 4) / 4 as grid_lon,
#                         ROUND(decimallatitude * 4) / 4 as grid_lat,
#                         EXTRACT(YEAR FROM eventdate) as observation_year,
#                         SUM(COALESCE(individualcount, 1)) as individual_count
#                     FROM `{os.getenv('GOOGLE_CLOUD_PROJECT')}.biodiversity.occurances_endangered_species_mammals`
#                     WHERE species = @species_name
#                         AND decimallongitude IS NOT NULL
#                         AND decimallatitude IS NOT NULL
#                         AND eventdate IS NOT NULL
#                         AND EXTRACT(YEAR FROM eventdate) > 2000
#                         {f"AND countrycode = @country_code" if country_code else ""}
#                     GROUP BY 
#                         grid_lon,
#                         grid_lat,
#                         observation_year
#                 ),
#                 forest_metrics AS (
#                     {forest_metrics}
#                 ),
#                 combined_data AS (
#                     SELECT 
#                         f.grid_lon,
#                         f.grid_lat,
#                         f.year,
#                         f.initial_tree_cover,
#                         f.remaining_tree_cover,
#                         f.cumulative_loss,
#                         COALESCE(s.individual_count, 0) as individual_count,
#                         s.observation_year
#                     FROM forest_metrics f
#                     LEFT JOIN species_counts s
#                         ON ROUND(f.grid_lon, 4) = ROUND(s.grid_lon, 4) 
#                         AND ROUND(f.grid_lat, 4) = ROUND(s.grid_lat, 4)
#                         AND f.year = s.observation_year  -- Match the exact year
#                 )
#                 SELECT
#                     observation_year,
#                     CORR(individual_count, remaining_tree_cover) as tree_cover_correlation,
#                     CORR(individual_count, cumulative_loss) as forest_loss_correlation,
#                     AVG(remaining_tree_cover) as mean_tree_cover,
#                     STDDEV(remaining_tree_cover) as std_tree_cover,
#                     AVG(cumulative_loss) as mean_forest_loss,
#                     STDDEV(cumulative_loss) as std_forest_loss,
#                     AVG(individual_count) as mean_individuals,
#                     STDDEV(individual_count) as std_individuals,
#                     COUNT(DISTINCT CONCAT(grid_lon, ',', grid_lat)) as total_cells,
#                     SUM(CASE WHEN individual_count > 0 THEN 1 ELSE 0 END) as cells_with_species
#                 FROM combined_data
#                 WHERE observation_year IS NOT NULL
#                 GROUP BY observation_year
#                 ORDER BY observation_year
#                 """

#                 query = self.build_query(query)  # Build the query with project ID
#                 self.logger.info("Executing single species query: %s", query)
                
#                 # Set up query parameters for single species analysis
#                 params = []
#                 if species_name:
#                     params.append(bigquery.ScalarQueryParameter("species_name", "STRING", species_name))
#                 if country_code:
#                     params.append(bigquery.ScalarQueryParameter("country_code", "STRING", country_code))
                
#                 job_config = bigquery.QueryJobConfig(query_parameters=params)
#                 df = client.query(query, job_config=job_config).to_dataframe()
                
#                 if df.empty:
#                     raise ValueError(f"No data found for species {species_name}")

#                 row = df.iloc[0]
#                 return {
#                     'forest_cover': {
#                         'mean': float(row['mean_tree_cover']),
#                         'std': float(row['std_tree_cover']),
#                         'correlation': float(row['tree_cover_correlation'] or 0),
#                         'p_value': 1.0
#                     },
#                     'forest_loss': {
#                         'mean': float(row['mean_forest_loss']),
#                         'std': float(row['std_forest_loss']),
#                         'correlation': float(row['forest_loss_correlation'] or 0),
#                         'p_value': 1.0
#                     },
#                     'total_cells': int(row['total_cells']),
#                     'cells_with_species': int(row['cells_with_species']),
#                     'mean_individuals': float(row['mean_individuals']),
#                     'std_individuals': float(row['std_individuals'])
#                 }

#             # Handle group analysis
#             if group_by not in ['conservation_status', 'country']:
#                 raise ValueError("group_by must be either 'conservation_status' or 'country'")

#             # Group analysis query
#             query = f"""
#             WITH species_forest AS (
#                 SELECT 
#                     sp.conservation_status,
#                     sp.species_name,
#                     sp.species_name_en,
#                     c.iso_a3 as country_code,
#                     c.name as country_name,
#                     ROUND(oc.decimallongitude * 4) / 4 as grid_lon,
#                     ROUND(oc.decimallatitude * 4) / 4 as grid_lat,
#                     EXTRACT(YEAR FROM oc.eventdate) as observation_year,
#                     SUM(COALESCE(oc.individualcount, 1)) as individual_count
#                 FROM `{os.getenv('GOOGLE_CLOUD_PROJECT')}.biodiversity.endangered_species` sp
#                 INNER JOIN (
#                     SELECT * 
#                     FROM `{os.getenv('GOOGLE_CLOUD_PROJECT')}.biodiversity.occurances_endangered_species_mammals`
#                     WHERE EXTRACT(YEAR FROM eventdate) > 2000  -- Only consider observations after 2000
#                 ) oc ON CONCAT(sp.genus_name, ' ', sp.species_name) = oc.species
#                 LEFT JOIN `{os.getenv('GOOGLE_CLOUD_PROJECT')}.biodiversity.countries` c
#                     ON ST_CONTAINS(c.geometry, ST_GEOGPOINT(oc.decimallongitude, oc.decimallatitude))
#                 WHERE 1=1
#                 {'AND c.iso_a3 = @country_code' if country_code else ''}
#                 {'AND sp.conservation_status = @conservation_status' if conservation_status else ''}
#                 GROUP BY 
#                     conservation_status,
#                     species_name,
#                     species_name_en,
#                     country_code,
#                     country_name,
#                     grid_lon,
#                     grid_lat,
#                     observation_year
#                 HAVING individual_count >= {min_observations}
#             ),
#             forest_metrics AS (
#                 {forest_metrics}
#             ),
#             combined_data AS (
#                 SELECT 
#                     s.species_name,
#                     s.species_name_en,
#                     s.conservation_status,
#                     s.country_code,
#                     s.country_name,
#                     s.grid_lon,
#                     s.grid_lat,
#                     s.individual_count,
#                     s.observation_year,
#                     f.initial_tree_cover,
#                     f.cumulative_loss,
#                     f.remaining_tree_cover
#                 FROM species_forest s
#                 INNER JOIN forest_metrics f
#                     ON ROUND(s.grid_lon, 4) = ROUND(f.grid_lon, 4) 
#                     AND ROUND(s.grid_lat, 4) = ROUND(f.grid_lat, 4)
#                     AND f.year = s.observation_year  -- Match the exact year
#             )
#             SELECT 
#                 species_name,
#                 species_name_en,
#                 conservation_status,
#                 country_code,
#                 country_name,
#                 CORR(individual_count, remaining_tree_cover) as tree_cover_correlation,
#                 CORR(individual_count, cumulative_loss) as forest_loss_correlation,
#                 COUNT(DISTINCT CONCAT(grid_lon, ',', grid_lat)) as number_of_grid_cells,
#                 SUM(individual_count) as total_individuals,
#                 AVG(individual_count) as avg_individuals_per_cell,
#                 AVG(initial_tree_cover) as avg_tree_cover,
#                 AVG(cumulative_loss) as avg_forest_loss,
#                 AVG(remaining_tree_cover) as forest_area_fraction
#             FROM combined_data
#             GROUP BY 
#                 species_name,
#                 species_name_en,
#                 conservation_status,
#                 country_code,
#                 country_name
#             HAVING number_of_grid_cells >= 5
#             ORDER BY species_name"""

#             # Set up query parameters
#             params = []
#             if species_name:
#                 params.append(bigquery.ScalarQueryParameter("species_name", "STRING", species_name))
#             if country_code:
#                 params.append(bigquery.ScalarQueryParameter("country_code", "STRING", country_code))
#             if conservation_status:
#                 params.append(bigquery.ScalarQueryParameter("conservation_status", "STRING", conservation_status))

#             job_config = bigquery.QueryJobConfig(query_parameters=params)

#             # Execute query and process results
#             query = self.build_query(query)  # Build the query with project ID
#             self.logger.info("Executing query: %s", query)
#             query_job = client.query(query, job_config=job_config)
#             query_results = list(query_job.result())  # Get all results

#             if not query_results:
#                 raise ValueError("No data found matching the specified criteria")

#             # Create a single result object
#             result = {
#                 'name': 'All Species',
#                 'total_species': 0,
#                 'correlations': [],
#                 'forest_metrics': {
#                     'forest_cover': {
#                         'mean': 0.0,
#                         'std': 0.0
#                     },
#                     'forest_loss': {
#                         'mean': 0.0,
#                         'std': 0.0
#                     },
#                     'forest_area': {
#                         'mean': 0.0,
#                         'std': 0.0
#                     },
#                     'sample_size': 0
#                 }
#             }

#             # Add all species correlation data
#             for row in query_results:
#                 result['correlations'].append({
#                     'species_name': row.species_name,
#                     'species_name_en': row.species_name_en,
#                     'conservation_status': row.conservation_status,
#                     'tree_cover_correlation': float(row.tree_cover_correlation or 0),
#                     'forest_loss_correlation': float(row.forest_loss_correlation or 0),
#                     'number_of_grid_cells': int(row.number_of_grid_cells),
#                     'total_individuals': int(row.total_individuals),
#                     'avg_individuals_per_cell': float(row.avg_individuals_per_cell),
#                     'avg_tree_cover': float(row.avg_tree_cover or 0),
#                     'avg_forest_loss': float(row.avg_forest_loss or 0),
#                     'forest_area_fraction': float(row.forest_area_fraction or 0)
#                 })

#             # Calculate aggregated metrics
#             correlations = result['correlations']
#             result['total_species'] = len(correlations)
            
#             # Calculate forest metrics
#             tree_covers = [sp['avg_tree_cover'] for sp in correlations if sp['avg_tree_cover'] is not None]
#             forest_losses = [sp['avg_forest_loss'] for sp in correlations if sp['avg_forest_loss'] is not None]
#             forest_areas = [sp['forest_area_fraction'] for sp in correlations if sp['forest_area_fraction'] is not None]
            
#             if tree_covers:
#                 result['forest_metrics']['forest_cover']['mean'] = float(sum(tree_covers) / len(tree_covers))
#                 result['forest_metrics']['forest_cover']['std'] = float(
#                     (sum((x - result['forest_metrics']['forest_cover']['mean']) ** 2 
#                     for x in tree_covers) / len(tree_covers)) ** 0.5
#                 )
            
#             if forest_losses:
#                 result['forest_metrics']['forest_loss']['mean'] = float(sum(forest_losses) / len(forest_losses))
#                 result['forest_metrics']['forest_loss']['std'] = float(
#                     (sum((x - result['forest_metrics']['forest_loss']['mean']) ** 2 
#                     for x in forest_losses) / len(forest_losses)) ** 0.5
#                 )

#             if forest_areas:
#                 result['forest_metrics']['forest_area']['mean'] = float(sum(forest_areas) / len(forest_areas))
#                 result['forest_metrics']['forest_area']['std'] = float(
#                     (sum((x - result['forest_metrics']['forest_area']['mean']) ** 2 
#                     for x in forest_areas) / len(forest_areas)) ** 0.5
#                 )

#             self.logger.info("Successfully processed %d results", len(query_results))
#             print(result)
#             return result
#         except Exception as e:
#             self.logger.error("Error calculating correlations: %s", str(e))
#             raise

 