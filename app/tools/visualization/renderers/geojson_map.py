"""
Renderer for GeoJSON map visualizations.
"""
from typing import Any, Dict, Optional
import time
import json
import pydeck as pdk
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class GeoJsonMapRenderer(BaseChartRenderer):
    """
    Renderer for GeoJSON map visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.GEOJSON_MAP]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a GeoJSON map visualization.

        Args:
            data: JSON string containing array of GeoJSON features
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            PyDeck deck object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            geojson_data = json.loads(data)
            bounds = self._get_bounds_from_geojson(geojson_data)

            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / len(bounds),
                    longitude=sum(coord[1] for coord in bounds) / len(bounds),
                    zoom=5,
                    pitch=30,
                )
            else:
                view_state = self.default_view_state

            # Extract features with IUCN category
            features = [
                {
                    "type": "Feature",
                    "geometry": item["geojson"],
                    "properties": {
                        "name": item["name"],
                        "category": item["category"],
                        "color": self._get_iucn_color(item["category"]),
                    },
                }
                for item in geojson_data
            ]
            geojson_layer = {"type": "FeatureCollection", "features": features}

            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend

            with col1:
                deck = pdk.Deck(
                    initial_view_state=view_state,
                    layers=[
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=geojson_layer,
                            get_fill_color="properties.color",
                            stroked=True,
                            filled=True,
                            pickable=True,
                            line_width_min_pixels=1,
                            get_line_color=[0, 0, 0],
                            opacity=0.8,
                            get_tooltip=["properties.name", "properties.category"],
                        )
                    ],
                    tooltip={"text": "Name: {name}\nIUCN Category: {category}"},
                )
                st.pydeck_chart(deck, height=700, key=f"geojson_map_{message_index}")

            with col2:
                st.markdown("### IUCN Categories")
                st.markdown("Only areas with an IUCN category are shown.")
                categories = {
                    "Ia": "Strict Nature Reserve",
                    "Ib": "Wilderness Area",
                    "II": "National Park",
                    "III": "Natural Monument",
                    "IV": "Habitat/Species Management",
                    "V": "Protected Landscape",
                    "VI": "Sustainable Use",
                }
                for cat, desc in categories.items():
                    color = self._get_iucn_color(cat)
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background:
                              rgb({color[0]},{color[1]},{color[2]});
                                      margin-right: 5px; border: 1px solid black;"></div>
                            <span><b>{cat}</b> - {desc}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            return deck

        except Exception as e:
            self.logger.error("Error creating geojson map: %s", str(e), exc_info=True)
            raise

    def _get_bounds_from_geojson(self, geojson_data):
        """Calculate bounds from GeoJSON features.
        Args:
            geojson_data (list): List of dictionaries containing GeoJSON features
        Returns:
            list: [[min_lat, min_lon], [max_lat, max_lon]] or None if invalid
        """
        try:
            all_coords = []
            for feature in geojson_data:
                geometry = feature["geojson"]
                if geometry["type"] == "Polygon":
                    coords = geometry["coordinates"][0]  # First ring of polygon
                    all_coords.extend(coords)
                elif geometry["type"] == "MultiPolygon":
                    for polygon in geometry["coordinates"]:
                        all_coords.extend(polygon[0])  # First ring of each polygon
            if not all_coords:
                return None
            # Convert to lat/lon pairs and find min/max
            lons, lats = zip(*all_coords)
            return [[min(lats), min(lons)], [max(lats), max(lons)]]
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error("Error calculating GeoJSON bounds: %s", str(e))
            return None

    def _get_iucn_color(self, category):
        """Return color for IUCN category."""
        # IUCN color scheme based on standard Protected Area colors
        iucn_colors = {
            "Ia": [0, 68, 27],  # Dark Green - Strict Nature Reserve
            "Ib": [0, 109, 44],  # Forest Green - Wilderness Area
            "II": [35, 139, 69],  # Green - National Park
            "III": [65, 171, 93],  # Light Green - Natural Monument
            "IV": [116, 196, 118],  # Pale Green - Habitat/Species Management
            "V": [161, 217, 155],  # Very Light Green - Protected Landscape
            "VI": [199, 233, 192],  # Lightest Green - Sustainable Use
            "Not Reported": [189, 189, 189],  # Gray
            "Not Applicable": [224, 224, 224],  # Light Gray
            "Not Assigned": [242, 242, 242],  # Very Light Gray
        }
        return iucn_colors.get(category, [150, 150, 150])  # Default gray for unknown