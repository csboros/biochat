"""
Renderer for tree visualization of species hierarchy.
"""
from typing import Any, Dict, Optional
import streamlit as st
try:
    import streamlit.components.v1 as components
except ImportError:
    components = None
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class TreeRenderer(BaseChartRenderer):
    """
    Renderer for tree visualization of species hierarchy.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.TREE_CHART]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a tree visualization.

        Args:
            data: Dictionary containing hierarchical species data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            None (displays visualization directly in Streamlit)
        """
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
            html_content = self.create_tree_html(data, width=950, height=800)
            html_content = html_content.replace('{data_placeholder}', str(data))
            # Set width to None to make it responsive
            components.html(html_content, height=1200, width=None)

        with col2:
            st.markdown("### Species Hierarchy")
            st.markdown("""
            This visualization shows the hierarchical relationship of endangered species:
            """)

            st.markdown("**Node Types:**")
            st.markdown("""
            <style>
            .color-box {
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
                border-radius: 50%;  /* This makes the boxes into circles */
                vertical-align: middle;
            }
            </style>
            <div style="margin-left: 10px;">
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #0066cc;"></span>
                    Class
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #00cccc;"></span>
                    Order
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: white; border: 1px solid black;"></span>
                    Families
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Updated color legend with Extinct status
            st.markdown("**Conservation Status:**")
            st.markdown("""
            <style>
            .color-box {
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
                border-radius: 50%;  /* This makes the boxes into circles */
                vertical-align: middle;
            }
            </style>
            <div style="margin-left: 10px;">
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #67000d;"></span>
                    Extinct (EX)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #d73027;"></span>
                    Critically Endangered (CR)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #fc8d59;"></span>
                    Endangered (EN)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #fee08b;"></span>
                    Vulnerable (VU)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #d9ef8b;"></span>
                    Near Threatened (NT)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #91cf60;"></span>
                    Least Concern (LC)
                </p>
                <p style="margin: 5px 0;">
                    <span class="color-box" style="background-color: #808080;"></span>
                    Data Deficient (DD)
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **Interaction:**
            - Hover over nodes to see details
            - White nodes represent families
            - Colored nodes represent species
            """)

    def create_tree_html(self, data, width=950, height=800):
        """Create the HTML/JavaScript code for D3.js radial cluster tree visualization."""
        return """
    <div id="tree-container" style="width: 100%;">
        <div id="tree-visualization" style="width: 100%; height: """ + str(height) + """px;">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                @media (prefers-color-scheme: dark) {
                    .link {
                        stroke: #A8C3BC !important;
                    }
                    .node-text {
                        fill: #A8C3BC !important;
                    }
                }
                @media (prefers-color-scheme: light) {
                    .link {
                        stroke: #353839 !important;
                    }
                    .node-text {
                        fill: #353839 !important;
                    }
                }
                #tree-container {
                    width: 100%;
                }
                #tree-visualization {
                    position: relative;
                    overflow: hidden;
                }
                #tree-svg {
                    width: 100%;
                    height: 100%;
                }
                .node circle {
                    stroke: #000;
                    stroke-width: 1px;
                }
                .link {
                    fill: none;
                    stroke-opacity: 0.6;
                    stroke-width: 1px;
                }
                .tooltip {
                    position: absolute;
                    padding: 8px;
                    background: rgba(0, 0, 0, 0.8);
                    color: #fff;
                    border-radius: 4px;
                    font: 12px sans-serif;
                    pointer-events: none;
                    z-index: 1000;
                }
            </style>
            <script>
                // Remove any existing SVG
                d3.select("#tree-visualization svg").remove();

                const height = """ + str(height) + """;
                const width = height;
                const radius = width / 2;

                // Create tooltip
                const tooltip = d3.select("#tree-visualization")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);

                // Create SVG with ID
                const svg = d3.select("#tree-visualization")
                    .append("svg")
                    .attr("id", "tree-svg")
                    .attr("viewBox", [-width / 2, -height / 2, width, height])
                    .style("font", "10px sans-serif");

                const data = {data_placeholder};

                // Color scale definition
                const color = d3.scaleOrdinal()
                    .domain(['Extinct', 'Critically Endangered', 'Endangered', 'Vulnerable',
                            'Near Threatened', 'Least Concern', 'Data Deficient'])
                    .range(['#67000d', '#d73027', '#fc8d59', '#fee08b',
                           '#d9ef8b', '#91cf60', '#808080']);

                // Define colors based on color scheme
                const linkColor = window.matchMedia('(prefers-color-scheme: dark)').matches ? '#A8C3BC' : '#353839';

                function updateVisualization() {
                    // Create cluster layout
                    const tree = d3.cluster()
                        .size([2 * Math.PI, radius - 100]);  // 100px padding for labels

                    // Generate tree data and filter out root
                    const root = d3.hierarchy(data);
                    const firstChild = root.children[0];  // Get the first child (class level)
                    firstChild.parent = null;  // Remove parent reference to make it the new root
                    tree(firstChild);  // Apply the layout to the new root

                    // Create links (no need to filter depth now)
                    svg.append("g")
                        .attr("fill", "none")
                        .selectAll("path")
                        .data(firstChild.links())
                        .join("path")
                        .attr("d", d3.linkRadial()
                            .angle(d => d.x)
                            .radius(d => d.y))
                        .style("stroke", linkColor)
                        .style("stroke-opacity", 0.9)
                        .style("stroke-width", "1px");

                    // Create nodes (no need to filter depth now)
                    const node = svg.append("g")
                        .selectAll("g")
                        .data(firstChild.descendants())
                        .join("g")
                        .attr("transform", d => `
                            rotate(${d.x * 180 / Math.PI - 90})
                            translate(${d.y},0)
                        `);

                    // Add circles to nodes (adjust depth numbers by -1)
                    node.append("circle")
                        .attr("r", d => {
                            if (d.depth === 1) return 8;      // Class (was depth 1)
                            if (d.depth === 2) return 6;      // Order (was depth 2)
                            if (d.depth === 3) return 4;      // Family (was depth 3)
                            return 3;                         // Species (was depth 4)
                        })
                        .style("fill", d => {
                            if (d.depth === 1) return "#0066cc";    // Class (was depth 1)
                            if (d.depth === 2) return "#00cccc";    // Order (was depth 2)
                            if (d.depth === 3) return "#fff";       // Family (was depth 3)
                            return color(d.data.status);            // Species (was depth 4)
                        })
                        .style("stroke", "#000")
                        .style("stroke-width", "1px");

                    // Add labels
                    node.append("text")
                        .attr("dy", "0.31em")
                        .attr("x", d => d.x < Math.PI === !d.children ? 6 : -6)
                        .attr("text-anchor", d => d.x < Math.PI === !d.children ? "start" : "end")
                        .attr("transform", d => d.x >= Math.PI ? "rotate(180)" : null)
                        .text(d => d.data.name)
                        .style("font-size", "10px")
                        .style("font-family", "sans-serif")
                        .style("fill", linkColor);

                    // Add hover effects
                    node.on("mouseover", function(event, d) {
                        // Highlight node
                        d3.select(this).select("circle")
                            .style("stroke-width", "2px");

                        // Show tooltip
                        let tooltipText = d.data.name;
                        if (d.data.species_name_en) {
                            tooltipText += `<br>${d.data.species_name_en}`;
                        }
                        if (d.data.status) {
                            tooltipText += `<br>${d.data.status}`;
                        }

                        tooltip.html(tooltipText)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 28) + "px")
                            .style("opacity", 0.9);
                    })
                    .on("mouseout", function() {
                        // Reset node style
                        d3.select(this).select("circle")
                            .style("stroke-width", "1px");

                        // Hide tooltip
                        tooltip.style("opacity", 0);
                    });
                }

                // Initialize visualization
                updateVisualization();
            </script>
        </div>
    </div>
    """
