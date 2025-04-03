"""Utility module for calculating alpha shapes from point distributions."""

import logging
from typing import  Optional, Tuple, List, Dict, Any
import numpy as np
from shapely.geometry import MultiPoint, Polygon
import ee

try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

# pylint: disable=broad-except
class AlphaShapeUtils:
    """Provides utilities for calculating alpha shapes from point distributions.

    This class handles both Python-based alpha shape calculation for visualization
    and Earth Engine-based alpha shape approximation for geospatial analysis.

    When used with forest metrics, the alpha shapes are used to sample from the Hansen dataset:
    - Forest cover: Tree canopy cover for year 2000, defined as percentage of canopy
      closure for all vegetation taller than 5m in height
    - Forest loss: Stand-replacement disturbance or change from forest to non-forest state.
      The 'lossyear' band indicates the year of loss (2001-2023, values 1-23) or 0 for no loss.
    """

    def __init__(self):
        """Initialize the AlphaShapeUtils."""
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def calculate_alpha_shape(self, points: np.ndarray, alpha: float = 0.5, eps: float = 2.0, min_samples: int = 3, avoid_overlaps: bool = True) -> Dict[str, Any]:
        """Calculate clustered alpha shapes for point distributions.

        Args:
            points: Array of [longitude, latitude] coordinates
            alpha: Alpha parameter (lower values create tighter shapes)
            eps: DBSCAN epsilon parameter - smaller values create smaller clusters
            min_samples: Minimum samples for DBSCAN - smaller values allow smaller clusters
            avoid_overlaps: Whether to merge overlapping clusters (default: True)

        Returns:
            GeoJSON representation of the alpha shapes
        """
        if Delaunay is None or DBSCAN is None:
            self.logger.error("Required dependencies missing. Install scipy and scikit-learn.")
            raise ImportError("Required dependencies missing. Install scipy and scikit-learn.")

        if len(points) < 4:
            # Not enough points for alpha shape, return convex hull
            return self._convert_hulls_to_geojson([MultiPoint(points).convex_hull])

        # Perform DBSCAN clustering with configurable parameters
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # Get unique cluster labels
        unique_labels = set(labels)
        if -1 in unique_labels:  # Remove noise label
            unique_labels.remove(-1)


        self.logger.info("Found %d clusters after subdivision", len(unique_labels))

        # Process each cluster
        hulls = []
        clusters = {}  # Dictionary to map labels to their corresponding hulls

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            cluster_points = points[labels == label]
            hull = self._process_single_cluster(cluster_points, alpha)
            if hull is not None:
                hulls.append(hull)
                clusters[label] = {
                    'hull': hull,
                    'points': cluster_points
                }

        # Check for overlapping hulls and merge them if requested
        if avoid_overlaps and hulls:
            merged_hulls = self._merge_overlapping_hulls(hulls, clusters, labels, points)
            self.logger.info("After merging overlaps: %d hulls", len(merged_hulls))
            result_hulls = merged_hulls
        else:
            result_hulls = hulls

        # Convert hulls to GeoJSON format
        return self._convert_hulls_to_geojson(result_hulls)

    def _process_clusters(
        self, points: np.ndarray, labels: np.ndarray, alpha: float
    ) -> List[Polygon]:
        """Process each cluster to create hulls."""
        hulls = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            cluster_points = points[labels == label]
            hull = self._process_single_cluster(cluster_points, alpha)
            if hull is not None:
                hulls.append(hull)
        return hulls

    def _process_single_cluster(
        self, cluster_points: np.ndarray, alpha: float
    ) -> Optional[Polygon]:
        """Process a single cluster to create a hull."""
        if len(cluster_points) < 4:
            # For very small clusters, create a convex hull with a buffer
            hull = MultiPoint(cluster_points).convex_hull
            # Add a small buffer to ensure points are fully contained
            return hull.buffer(0.05, resolution=16)
        try:
            # Create an alpha shape
            hull = self._create_alpha_shape(cluster_points, alpha)
            # Add a small buffer to ensure points are fully contained
            return hull.buffer(0.02, resolution=16)
        except Exception as e:  # pylint: disable=broad-except
            m = MultiPoint(cluster_points)
            return m.convex_hull.buffer(0.05, resolution=16)

    def _create_alpha_shape(self, points: np.ndarray, alpha: float) -> Polygon:
        """Create alpha shape from points."""
        tri = Delaunay(points)
        edges = set()
        edge_points = []

        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            circum_r = self._calculate_circumradius(pa, pb, pc)
            if circum_r < 1.0 / alpha:
                self._add_edge(edges, edge_points, (points, ia, ib))
                self._add_edge(edges, edge_points, (points, ib, ic))
                self._add_edge(edges, edge_points, (points, ic, ia))

        m = MultiPoint(points)
        polygon = Polygon(m.convex_hull)
        if polygon.is_valid:
            return polygon.buffer(0.02, resolution=16)
        return m.convex_hull.buffer(0.05, resolution=16)

    def _calculate_circumradius(self, pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
        """Calculate circumradius of triangle."""
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return a * b * c / (4.0 * area) if area > 0 else float("inf")

    def _add_edge(self, edges: set, edge_points: list, point_data: tuple) -> None:
        """Helper method to add edges."""
        points, i, j = point_data
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(points[[i, j]])

    def _convert_hulls_to_geojson(self, hulls: List[Polygon]) -> Dict[str, Any]:
        """Convert hulls to GeoJSON format."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[list(p) for p in hull.exterior.coords]]
                    for hull in hulls
                    if hasattr(hull, "exterior")
                ],
            },
        }

    def create_ee_alpha_shape(
        self,
        observations: List[Dict[str, Any]],
        alpha: float = 0.3,
        max_shapes: int = 5,
        eps: float = 2.0,
        min_samples: int = 3,
        avoid_overlaps: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create alpha shapes from observations and convert them to Earth Engine FeatureCollection.

        Args:
            observations: List of observation dictionaries with latitude and longitude.
            alpha: Alpha parameter for the alpha shape. Lower values create tighter shapes.
            max_shapes: Maximum number of alpha shapes to create.
            eps: DBSCAN eps parameter.
            min_samples: DBSCAN min_samples parameter.
            avoid_overlaps: Whether to avoid overlaps between alpha shapes.

        Returns:
            Tuple containing:
            - List of alpha shape dictionaries.
            - List of Earth Engine feature dictionaries.
        """

        if len(observations) < 4:
            # For few points, handle specially
            if len(observations) > 0:
                # Create a simple convex hull
                points = [(obs["decimallongitude"], obs["decimallatitude"])
                         for obs in observations]
                mp = MultiPoint(points)
                hull = mp.convex_hull

                # Check the geometry type - it could be a Point if all points are identical
                if hull.geom_type == 'Point':
                    # For a point, create a small buffer to make it a polygon
                    self.logger.warning("Convex hull resulted in a Point. Creating a small buffer.")
                    hull = hull.buffer(0.01)  # Small buffer around the point

                hull_coords = [list(p) for p in hull.exterior.coords]

                # Create Earth Engine feature
                ee_geom = ee.Geometry.Polygon(hull_coords)
                years = [obs["observation_year"] for obs in observations]
                representative_year = max(years) if years else 2000
                total_individuals = sum([obs["individual_count"] for obs in observations])

                ee_feature = ee.Feature(ee_geom, {
                    'year': representative_year,
                    'individual_count': total_individuals,
                    'num_observations': len(observations)
                })

                alpha_shape = {
                    'type': 'Polygon',
                    'coordinates': [hull_coords],
                    'properties': {
                        'year': representative_year,
                        'num_observations': len(observations),
                        'total_individuals': total_individuals
                    }
                }

                return [alpha_shape], [ee_feature]
            return [], []

        # Call calculate_alpha_shape to get the same shapes -
        points = np.array([
            [obs["decimallongitude"], obs["decimallatitude"]]
            for obs in observations
        ])
        geojson_result = self.calculate_alpha_shape(
            points, alpha, eps, min_samples, avoid_overlaps
        )

        # Extract the polygons from the GeoJSON
        all_alpha_shapes = []
        ee_features = []

        if (geojson_result and
            'geometry' in geojson_result and
            'coordinates' in geojson_result['geometry']):
            # Perform clustering as in calculate_alpha_shape to assign observations
            # to the proper hulls
            # First we need to re-do the clustering
            points = np.array([
                [obs["decimallongitude"], obs["decimallatitude"]]
                for obs in observations
            ])
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_

            # Perform subdivision as in calculate_alpha_shape
            unique_labels = set(labels) - {-1}

            # If we have too few clusters, try with even smaller eps
            if len(unique_labels) < 2 and len(observations) > 20:
                points = np.array([[obs["decimallongitude"], obs["decimallatitude"]]
                                 for obs in observations])
                clustering = DBSCAN(eps=eps/2, min_samples=min_samples).fit(points)
                labels = clustering.labels_
                unique_labels = set(labels) - {-1}

            # Handle large clusters subdivision
            if len(unique_labels) > 0:
                new_labels = labels.copy()
                next_label = max(unique_labels) + 1 if unique_labels else 0

                for label in unique_labels:
                    cluster_points = np.array([
                        [obs["decimallongitude"], obs["decimallatitude"]]
                        for obs in observations if obs["observation_year"] == label
                    ])
                    if len(cluster_points) > 30:
                        sub_clustering = DBSCAN(eps=eps/3, min_samples=2).fit(cluster_points)
                        sub_labels = sub_clustering.labels_
                        sub_unique_labels = set(sub_labels) - {-1}

                        if len(sub_unique_labels) > 1:
                            idx = 0
                            for i, l in enumerate(labels):
                                if l == label:
                                    sub_label = sub_labels[idx]
                                    if sub_label != -1:
                                        new_labels[i] = next_label + sub_label
                                    idx += 1

                labels = new_labels
                unique_labels = set(labels) - {-1}

            # Process the merged hulls from GeoJSON
            # The difference here is that we're accessing the MultiPolygon coordinates differently
            # because we're dealing with merged shapes now
            polygons = geojson_result['geometry']['coordinates']

            # Determine which observations fall into which shapes
            # We'll need to use Shapely to check point containment in each hull
            shapely_polygons = []
            for polygon_coords in polygons:
                if isinstance(polygon_coords[0][0], list):
                    # This is a valid polygon with at least one ring
                    hull_coords = polygon_coords[0]
                else:
                    # This is a simpler polygon structure
                    hull_coords = polygon_coords
                shapely_polygons.append(Polygon(hull_coords))

            # Assign observations to the shapes they fall within
            shape_observations = [[] for _ in range(len(shapely_polygons))]
            for i, obs in enumerate(observations):
                point = (obs["decimallongitude"], obs["decimallatitude"])
                assigned = False

                # First check if the point is contained within any polygon
                for j, polygon in enumerate(shapely_polygons):
                    point_geom = MultiPoint([point])
                    if polygon.contains(point_geom) or polygon.intersects(point_geom):
                        shape_observations[j].append(obs)
                        assigned = True
                        break

                # If not assigned, find the nearest polygon
                if not assigned:
                    point_geom = MultiPoint([point])
                    distances = [polygon.distance(point_geom) for polygon in shapely_polygons]
                    if distances:  # Make sure we have polygons
                        nearest_polygon_index = distances.index(min(distances))
                        shape_observations[nearest_polygon_index].append(obs)

            # Check if any polygons have no assigned observations
            empty_indices = [i for i, obs_list in enumerate(shape_observations) if not obs_list]
            for i in sorted(empty_indices, reverse=True):
                del shape_observations[i]
                del shapely_polygons[i]

            # Limit to max_shapes - prefer shapes with more observations
            if len(shapely_polygons) > max_shapes:
                shapes_with_obs = [
                    (i, len(obs_list)) for i, obs_list in enumerate(shape_observations)
                ]
                shapes_with_obs.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [i for i, _ in shapes_with_obs[:max_shapes]]
                selected_indices.sort()  # Sort to maintain original order
                shapely_polygons = [shapely_polygons[i] for i in selected_indices]
                shape_observations = [shape_observations[i] for i in selected_indices]

            # Create Earth Engine features for each shape
            for i, (polygon, obs_list) in enumerate(zip(shapely_polygons, shape_observations)):
                if not obs_list:
                    continue  # Skip shapes with no observations

                # Extract the exterior coordinates
                hull_coords = [list(p) for p in polygon.exterior.coords]

                # Calculate properties for this shape
                years = [obs["observation_year"] for obs in obs_list]
                representative_year = max(years) if years else 2000
                total_individuals = sum([obs["individual_count"] for obs in obs_list])

                try:
                    # Create Earth Engine geometry and feature
                    ee_geom = ee.Geometry.Polygon(hull_coords)
                    ee_feature = ee.Feature(ee_geom, {
                        'year': representative_year,
                        'individual_count': total_individuals,
                        'num_observations': len(obs_list)
                    })

                    ee_features.append(ee_feature)

                    # Store alpha shape for visualization
                    alpha_shape = {
                        'type': 'Polygon',
                        'coordinates': [hull_coords],
                        'properties': {
                            'year': representative_year,
                            'num_observations': len(obs_list),
                            'total_individuals': total_individuals
                        }
                    }

                    all_alpha_shapes.append(alpha_shape)
                except Exception as e:
                    self.logger.warning("Error creating Earth Engine feature: %s", str(e))

        # If we couldn't extract any shapes, fall back to a convex hull of all points
        if not all_alpha_shapes and len(observations) > 0:
            self.logger.warning("Failed to extract alpha shapes, falling back to convex hull.")
            points = [(obs["decimallongitude"], obs["decimallatitude"])
                     for obs in observations]
            mp = MultiPoint(points)
            hull = mp.convex_hull

            if hull.geom_type == 'Point':
                hull = hull.buffer(0.01)

            hull_coords = [list(p) for p in hull.exterior.coords]

            # Create Earth Engine feature
            ee_geom = ee.Geometry.Polygon(hull_coords)
            years = [obs["observation_year"] for obs in observations]
            representative_year = max(years) if years else 2000
            total_individuals = sum([obs["individual_count"] for obs in observations])

            ee_feature = ee.Feature(ee_geom, {
                'year': representative_year,
                'individual_count': total_individuals,
                'num_observations': len(observations)
            })

            alpha_shape = {
                'type': 'Polygon',
                'coordinates': [hull_coords],
                'properties': {
                    'year': representative_year,
                    'num_observations': len(observations),
                    'total_individuals': total_individuals
                }
            }

            all_alpha_shapes = [alpha_shape]
            ee_features = [ee_feature]

        self.logger.info("Created %d EE alpha shapes", len(all_alpha_shapes))

        # Process the final shapes
        final_shapes = all_alpha_shapes
        ee_features = []

        for shape in final_shapes:
            try:
                # Extract coordinates from the shape
                if 'type' in shape and shape['type'] == 'Polygon' and 'coordinates' in shape:
                    hull_coords = shape['coordinates'][0]

                    # Create Earth Engine geometry and feature
                    ee_geom = ee.Geometry.Polygon(hull_coords)
                    ee_feature = ee.Feature(ee_geom, {
                        'year': shape.get('properties', {}).get('year', 2000),
                        'individual_count': shape.get('properties', {}).get('total_individuals', 0),
                        'num_observations': shape.get('properties', {}).get('num_observations', 0)
                    })

                    ee_features.append(ee_feature)
            except Exception as e:
                self.logger.warning("Error creating Earth Engine feature: %s", str(e))

        return final_shapes, ee_features

    def _merge_overlapping_hulls(self, hulls: List[Polygon], clusters: Dict, labels: np.ndarray, all_points: np.ndarray) -> List[Polygon]:
        """Merges overlapping hulls to avoid cluster overlaps while ensuring all observations are covered.

        Args:
            hulls: List of hull polygons
            clusters: Dictionary mapping cluster labels to hulls and points
            labels: Array of cluster labels for all points
            all_points: Array of all point coordinates

        Returns:
            List of merged hulls with no overlaps and full observation coverage
        """
        if not hulls:
            return []

        # First, make sure all valid observations (non-noise) are covered by at least one hull
        covered_points_mask = np.zeros(len(all_points), dtype=bool)

        # Create a mapping from point index to cluster label
        point_labels = {}
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            for idx in np.where(labels == label)[0]:
                point_labels[idx] = label

        # Check which points are covered by existing hulls
        for i, point in enumerate(all_points):
            if i not in point_labels:  # Skip noise points
                continue

            point_geom = MultiPoint([point])
            for hull in hulls:
                if hull.contains(point_geom) or hull.intersects(point_geom):
                    covered_points_mask[i] = True
                    break

        # If any valid points are not covered, expand the nearest hull to include them
        uncovered_points = all_points[~covered_points_mask & np.array([i in point_labels for i in range(len(all_points))])]

        if len(uncovered_points) > 0:
            self.logger.info("Found %d uncovered points, expanding hulls to include them", len(uncovered_points))

            # For each uncovered point, find the nearest hull and expand it
            for point in uncovered_points:
                point_geom = MultiPoint([point])
                min_dist = float('inf')
                nearest_hull_idx = -1

                # Find the nearest hull
                for i, hull in enumerate(hulls):
                    dist = hull.distance(point_geom)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_hull_idx = i

                if nearest_hull_idx >= 0:
                    # Create a small buffer around the point and union with the nearest hull
                    point_buffer = point_geom.buffer(0.05)
                    hulls[nearest_hull_idx] = hulls[nearest_hull_idx].union(point_buffer)

        # Create a copy of hulls to work with for merging
        merged_hulls = hulls.copy()

        # Keep track of which hulls have been merged
        merged_indices = set()
        i = 0

        # Merge overlapping hulls
        while i < len(merged_hulls):
            if i in merged_indices:
                i += 1
                continue

            hull_i = merged_hulls[i]
            merged_with_current = False

            j = i + 1
            while j < len(merged_hulls):
                if j in merged_indices:
                    j += 1
                    continue

                hull_j = merged_hulls[j]

                # Check if hulls intersect
                if hull_i.intersects(hull_j):
                    self.logger.info("Found overlapping hulls at indices %d and %d", i, j)

                    # Merge the hulls using union
                    try:
                        merged_hull = hull_i.union(hull_j)

                        # Sometimes union results in a MultiPolygon,
                        # in this case take the largest polygon
                        if merged_hull.geom_type == 'MultiPolygon':
                            areas = [p.area for p in merged_hull.geoms]
                            merged_hull = merged_hull.geoms[areas.index(max(areas))]

                        # Update the current hull with the merged one
                        merged_hulls[i] = merged_hull
                        merged_indices.add(j)
                        merged_with_current = True
                    except Exception as e:
                        self.logger.warning("Failed to merge hulls: %s", str(e))

                j += 1

            # If we merged the current hull with others, we need to check again with the updated hull
            if not merged_with_current:
                i += 1

        # Remove merged hulls
        result = [hull for idx, hull in enumerate(merged_hulls) if idx not in merged_indices]

        # Final check: ensure all observations are covered by at least one hull
        covered_points_mask = np.zeros(len(all_points), dtype=bool)

        for i, point in enumerate(all_points):
            if i not in point_labels:  # Skip noise points
                continue

            point_geom = MultiPoint([point])
            for hull in result:
                if hull.contains(point_geom) or hull.intersects(point_geom):
                    covered_points_mask[i] = True
                    break

        # For any remaining uncovered points, create a new hull for each cluster of points
        uncovered_mask = np.array([i in point_labels for i in range(len(all_points))])
        uncovered_indices = np.where(~covered_points_mask & uncovered_mask)[0]

        if len(uncovered_indices) > 0:
            self.logger.info("After merging, found %d uncovered points, creating new hulls", len(uncovered_indices))

            # Group uncovered points by their original cluster
            uncovered_clusters = {}
            for idx in uncovered_indices:
                label = point_labels[idx]
                if label not in uncovered_clusters:
                    uncovered_clusters[label] = []
                uncovered_clusters[label].append(all_points[idx])

            # Create a new hull for each uncovered cluster
            for label, points in uncovered_clusters.items():
                if len(points) < 4:
                    # For small clusters, use a convex hull with buffer
                    hull = MultiPoint(points).convex_hull.buffer(0.05)
                else:
                    # For larger clusters, use an alpha shape
                    try:
                        hull = self._create_alpha_shape(np.array(points), 0.5)
                    except Exception:
                        hull = MultiPoint(points).convex_hull.buffer(0.05)

                # Add the new hull
                result.append(hull)

        # Log the results
        self.logger.info("Final result: %d hulls", len(result))
        return result
