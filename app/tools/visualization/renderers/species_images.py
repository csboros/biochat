"""
Renderer for species images visualization.
"""
from typing import Any, Dict, Optional
import time
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class SpeciesImagesRenderer(BaseChartRenderer):
    """
    Renderer for species images visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.SPECIES_IMAGES]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a species images visualization.

        Args:
            data: Dictionary containing species images data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            None (displays images directly in Streamlit)
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())

            if data["image_count"] > 0:
                st.subheader(f"Images of {data['species']}", key=f"species_images_{message_index}")
                cols = st.columns(min(data["image_count"], 3))  # Up to 3 columns

                for idx, img in enumerate(data["images"]):
                    with cols[idx % 3]:
                        try:
                            st.image(
                                img["url"],
                                use_column_width=True,
                                caption=f"Source: {img['publisher']}\nBy: {img['creator']}\n"
                                f"License: {img['license']}"
                            )
                        except Exception:  # pylint: disable=broad-except
                            st.warning("Could not load image")
            else:
                st.info(f"No images found for {data['species']}")

            return None

        except Exception as e:
            self.logger.error("Error displaying species images: %s", str(e), exc_info=True)
            raise
