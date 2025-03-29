"""Renderer for habitat analysis visualization."""

import ee
import folium
from typing import Dict, Any

class HabitatViz:
    """Renders habitat analysis results on a map."""

    def __init__(self):
        self.land_cover_classes = {
            111: "Closed forest, evergreen needle leaf",
            112: "Closed forest, evergreen broad leaf",
            113: "Closed forest, deciduous needle leaf",
            114: "Closed forest, deciduous broad leaf",
            115: "Closed forest, mixed",
            116: "Closed forest, unknown type",
            121: "Open forest, evergreen needle leaf",
            122: "Open forest, evergreen broad leaf",
            123: "Open forest, deciduous needle leaf",
            124: "Open forest, deciduous broad leaf",
            125: "Open forest, mixed",
            126: "Open forest, unknown type",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Urban/built up",
            60: "Bare/sparse vegetation",
            70: "Snow and ice",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            100: "Moss and lichen",
            200: "Open sea"
        }

    def render(
        self,
        landcover: ee.Image,
        species_points: ee.FeatureCollection,
        analysis_results: Dict[str, Any],
        center: Dict[str, float]
    ) -> folium.Map:
        """
        Render habitat analysis results on a map.

        Args:
            landcover: Earth Engine image of land cover data
            species_points: FeatureCollection of species observation points
            analysis_results: Dictionary containing analysis results
            center: Dictionary with 'lat' and 'lon' for map center

        Returns:
            folium.Map object with visualization
        """
        # Create base map
        m = folium.Map(
            location=[center['lat'], center['lon']],
            zoom_start=8
        )

        # Add land cover layer
        landcover_vis = {
            'min': 0,
            'max': 200,
            'palette': [
                '#1a9850', '#91cf60', '#d9ef8b', '#ffffbf', '#fee08b',
                '#fc8d59', '#e41a1c', '#000000', '#ffffff', '#808080'
            ]
        }

        # Add land cover to map
        folium.raster_layers.TileLayer(
            tiles=landcover.select('discrete_classification').getMapId(landcover_vis)['tile_fetcher'].url_format,
            attr='Land Cover',
            name='Land Cover Types'
        ).add_to(m)

        # Add species points
        species_points_vis = {
            'color': 'red',
            'fillColor': 'red',
            'fillOpacity': 0.7,
            'radius': 5
        }

        folium.GeoJson(
            data=species_points.getInfo(),
            name='Species Observations',
            style_function=lambda x: species_points_vis
        ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add legend
        legend_html = self._create_legend()
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def _create_legend(self) -> str:
        """Create HTML legend for land cover types."""
        legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <h4>Land Cover Types</h4>
        '''

        for code, name in self.land_cover_classes.items():
            legend_html += f'<p style="margin: 5px 0;"><span style="color: {self._get_color_for_code(code)}">■</span> {name}</p>'

        legend_html += '</div>'
        return legend_html

    def _get_color_for_code(self, code: int) -> str:
        """Get color for specific land cover code."""
        # Define colors for different land cover types
        color_map = {
            # Forest types (green shades)
            111: '#1a9850', 112: '#91cf60', 113: '#d9ef8b',
            114: '#ffffbf', 115: '#fee08b', 116: '#fc8d59',
            121: '#1a9850', 122: '#91cf60', 123: '#d9ef8b',
            124: '#ffffbf', 125: '#fee08b', 126: '#fc8d59',
            # Other types
            20: '#c2a5cf',  # Shrubland
            30: '#ffff00',  # Grassland
            40: '#ffd700',  # Cropland
            50: '#ff0000',  # Urban
            60: '#8b4513',  # Bare
            70: '#ffffff',  # Snow
            80: '#0000ff',  # Water
            90: '#00ff00',  # Wetland
            100: '#808080',  # Moss
            200: '#000080'   # Sea
        }
        return color_map.get(code, '#808080')  # Default gray