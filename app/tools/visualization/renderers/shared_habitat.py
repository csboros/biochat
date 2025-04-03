"""
Renderer for shared habitat network visualization using D3.js.
"""
from typing import Any, Dict, Optional
import json
import streamlit as st
try:
    import streamlit.components.v1 as components
except ImportError:
    components = None
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class SharedHabitatRenderer(BaseChartRenderer):
    """
    Renderer for shared habitat network visualization using D3.js.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.SPECIES_SHARED_HABITAT]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a shared habitat network visualization.

        Args:
            data: Dictionary containing species correlation data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            None (displays visualization directly in Streamlit)
        """
        try:
            # Add responsive CSS
            st.markdown("""
                <style>
                    .element-container {
                        width: 100% !important;
                    }
                    iframe {
                        width: 100% !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])

            with col1:
                # Create network data structure
                nodes = []
                nodes_set = set()
                # Create nodes
                for correlation in data.get("correlations", []):
                    if correlation["species_1"] not in nodes_set:
                        nodes.append({
                            "id": correlation["species_1"],
                            "name": correlation["species_1"],
                            "english_name": correlation["species_1_en"],
                            "status": correlation["species_1_status"]
                        })
                        nodes_set.add(correlation["species_1"])
                    if correlation["species_2"] not in nodes_set:
                        nodes.append({
                            "id": correlation["species_2"],
                            "name": correlation["species_2"],
                            "english_name": correlation["species_2_en"],
                            "status": correlation["species_2_status"]
                        })
                        nodes_set.add(correlation["species_2"])

                # Create links
                links = []
                for correlation in data.get("correlations", []):
                    links.append({
                        "source": correlation["species_1"],
                        "target": correlation["species_2"],
                        "correlation": correlation["correlation_coefficient"],
                        "overlapping_cells": correlation["overlapping_cells"]
                    })

                # Create network data structure
                network_data = {
                    "nodes": nodes,
                    "links": links
                }

                html_content = self._create_shared_habitat_html(network_data)
                components.html(html_content, height=1000)

            with col2:
                st.markdown("### Network Legend")
                st.markdown("""
                    **Nodes (Species)**
                    - Size: Fixed
                    - Color: Conservation status

                    **Links (Connections)**
                    - Width: Correlation strength (Only correlations > 0.2 are shown)

                """)

                # Add conservation status legend
                st.markdown("### Conservation Status")
                status_colors = {
                    'Extinct': '#67000d',
                    'Critically Endangered': '#d73027',
                    'Endangered': '#fc8d59',
                    'Vulnerable': '#fee08b',
                    'Near Threatened': '#d9ef8b',
                    'Least Concern': '#91cf60',
                    'Data Deficient': '#808080'
                }

                for status, color in status_colors.items():
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px;
                                      background-color: {color};
                                      margin-right: 5px;
                                      border-radius: 50%;
                                      border: 2px solid black;">
                            </div>
                            <span>{status}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Add statistics
                st.markdown("### Network Statistics")
                avg_correlation = sum(link['correlation'] for link in links) / len(links) if links else 0
                st.markdown(f"""
                    - Total Species: {len(nodes)}
                    - Total Connections: {len(links)}
                    - Average Correlation: {avg_correlation:.3f}
                """)

            return None

        except Exception as e:
            self.logger.error("Error displaying shared habitat network: %s", str(e), exc_info=True)
            raise

    def _create_shared_habitat_html(self, data: Any, width: int = 950, height: int = 1000) -> str:
        """Create the HTML/JavaScript code for D3.js shared habitat network visualization."""
        return """
        <div id="network" style="width: 100%; height: """ + str(height) + """px; background-color: #ffffff;">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
                const networkData = """ + json.dumps(data) + """;

                // Set up SVG
                const container = document.getElementById("network");
                const width = container.clientWidth;
                const height = container.clientHeight;

                const svg = d3.select("#network")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);

                // Add a group for zoom transformation
                const g = svg.append("g");

                // Color scale for conservation status
                const statusColorScale = d3.scaleOrdinal()
                    .domain(['Extinct', 'Critically Endangered', 'Endangered', 'Vulnerable',
                            'Near Threatened', 'Least Concern', 'Data Deficient'])
                    .range([
                        '#67000d',  // Extinct - Very Dark Red
                        '#d73027',  // Critically Endangered - Red
                        '#fc8d59',  // Endangered - Orange
                        '#fee08b',  // Vulnerable - Yellow
                        '#d9ef8b',  // Near Threatened - Light Green
                        '#91cf60',  // Least Concern - Green
                        '#808080'   // Data Deficient - Gray
                    ]);

                // Set up forces
                const simulation = d3.forceSimulation(networkData.nodes)
                    .force("link", d3.forceLink(networkData.links)
                        .id(d => d.id)
                        .distance(d => (1 - Math.abs(d.correlation)) * 200)
                        .strength(d => Math.abs(d.correlation)))
                    .force("charge", d3.forceManyBody().strength(-1000))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide().radius(50));

                // Create links
                const link = g.append("g")
                    .selectAll("line")
                    .data(networkData.links)
                    .join("line")
                    .style("stroke", d => d.correlation > 0 ? "#4292c6" : "#ef3b2c")
                    .style("stroke-width", d => Math.abs(d.correlation) * 5)
                    .style("stroke-opacity", 0.6);

                // Create nodes
                const node = g.append("g")
                    .selectAll("g")
                    .data(networkData.nodes)
                    .join("g")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));

                // Add circles to nodes
                node.append("circle")
                    .attr("r", 15)
                    .style("fill", d => statusColorScale(d.status))
                    .style("stroke", "#000")
                    .style("stroke-width", "1px");

                // Add labels to nodes
                node.append("text")
                    .attr("dx", 18)
                    .attr("dy", ".35em")
                    .text(d => d.name)
                    .style("fill", "#333")
                    .style("font-size", "12px")
                    .style("font-weight", "bold");

                // Create tooltip div
                const tooltip = d3.select("#network")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("position", "absolute")
                    .style("padding", "8px")
                    .style("background", "rgba(0, 0, 0, 0.8)")
                    .style("color", "#fff")
                    .style("border-radius", "4px")
                    .style("pointer-events", "none")
                    .style("font", "12px sans-serif")
                    .style("opacity", 0)
                    .style("z-index", 1000);

                // Add hover interactions
                node.on("mouseover", function(event, d) {
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`<strong>Species:</strong> ${d.name}<br>` +
                               `<strong>English Name:</strong> ${d.english_name}<br>` +
                               `<strong>Status:</strong> ${d.status}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                })
                .on("mouseout", function() {
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                });

                // Add zoom behavior
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on("zoom", zoomed);

                svg.call(zoom);

                function zoomed(event) {
                    g.attr("transform", event.transform);
                }

                // Update positions on each tick
                simulation.on("tick", () => {
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);

                    node
                        .attr("transform", d => `translate(${d.x},${d.y})`);
                });

                // Drag functions
                function dragstarted(event) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }

                function dragged(event) {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }

                function dragended(event) {
                    if (!event.active) simulation.alphaTarget(0);
                    // Release the node's fixed position
                    event.subject.fx = null;
                    event.subject.fy = null;
                }
            </script>
        </div>
        """
