"""
Renderer for force-directed graph visualization using D3.js.
"""
from typing import Any, Dict, Optional
import time
import streamlit as st
try:
    import streamlit.components.v1 as components
except ImportError:
    components = None
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class ForceDirectedGraphRenderer(BaseChartRenderer):
    """
    Renderer for force-directed graph visualization using D3.js.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.FORCE_DIRECTED_GRAPH]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a force-directed graph visualization.

        Args:
            data: Dictionary containing hierarchical species data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            None (displays visualization directly in Streamlit)
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())

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
                html_content = self._create_force_html(data)
                html_content = html_content.replace('{data_placeholder}', str(data))
                # Set width to None to make it responsive
                components.html(html_content, height=900, width=None)

            with col2:
                st.markdown("### Species Network")
                st.markdown("""
                This visualization shows the relationships between species in a force-directed layout:
                """)

                # Updated color legend with Class, Order and Conservation Status
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

                st.markdown("**Conservation Status:**")
                st.markdown("""
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
                - Drag nodes to rearrange the network
                - Hover over nodes to see details
                - Network will automatically adjust positions
                """)

            return None

        except Exception as e:
            self.logger.error("Error displaying force-directed graph: %s", str(e), exc_info=True)
            raise

    def _create_force_html(self, data: Any, width: int = 950, height: int = 800) -> str:
        """Create the HTML/JavaScript code for D3.js force-directed visualization."""
        return """
        <div id="force-visualization" style="width: 100%; height: """ + str(height) + """px;">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                #force-visualization {
                    position: relative;
                    overflow: hidden;
                }
                svg {
                    width: 100% !important;
                    height: 100% !important;
                }
                .node circle {
                    stroke: #000;
                    stroke-width: 1px;
                }
                .link {
                    fill: none;
                    stroke: #999;
                    stroke-opacity: 0.6;
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
                function getContainerWidth() {
                    return document.getElementById('force-visualization').getBoundingClientRect().width;
                }

                const height = """ + str(height) + """;
                let width = getContainerWidth();

                // Create tooltip
                const tooltip = d3.select("#force-visualization")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);

                const svg = d3.select("#force-visualization")
                    .append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%");

                const data = {data_placeholder};

                // Process data into nodes and links
                const nodes = [];
                const links = [];

                // Process classes, orders, families, and species (skip root node)
                data.children.forEach((classGroup) => {
                    // Add class node
                    nodes.push({
                        id: classGroup.name,
                        group: "class",
                        radius: 10
                    });

                    // Process orders
                    classGroup.children.forEach((order) => {
                        // Add order node
                        nodes.push({
                            id: order.name,
                            group: "order",
                            radius: 8
                        });
                        links.push({
                            source: classGroup.name,
                            target: order.name
                        });

                        // Process families
                        order.children.forEach((family) => {
                            nodes.push({
                                id: family.name,
                                group: "family",
                                radius: 6
                            });
                            links.push({
                                source: order.name,
                                target: family.name
                            });

                            // Process species
                            family.children.forEach((species) => {
                                // Create unique ID for species
                                const speciesId = `${species.name}`;
                                // Add species with English name if available
                                const species_info = {
                                    id: speciesId,
                                    displayName: species.name,
                                    group: "species",
                                    status: species.status,
                                    radius: 4
                                }
                                if (species.species_name_en) {  // JavaScript syntax for checking property existence
                                    species_info['species_name_en'] = species.species_name_en;
                                }

                                nodes.push(species_info)
                                links.push({
                                    source: family.name,
                                    target: speciesId
                                });
                            });
                        });
                    });
                });

                // Update radius based on node count
                const nodeCount = nodes.length;
                const radiusScale = nodeCount < 50 ? 2.5 :
                                  nodeCount < 100 ? 1.5 :
                                  1;
                nodes.forEach(node => {
                    node.radius = node.radius * radiusScale;
                });

                // Color scale for conservation status
                const color = d3.scaleOrdinal()
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

                // Update force simulation parameters
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links)
                        .id(d => d.id)
                        .distance(d => {
                            const nodeCount = nodes.length;
                            const scaleFactor = nodeCount < 50 ? 10 :
                                              nodeCount < 100 ? 5 :
                                              1;

                            if (d.source.group === "class") return 30 * scaleFactor;
                            if (d.source.group === "order") return 10 * scaleFactor;
                            if (d.source.group === "family") return 20 * scaleFactor;
                            return 15 * scaleFactor;
                        }))
                    .force("charge", d3.forceManyBody()
                        .strength(d => {
                            if (d.group === "class") return -400;
                            if (d.group === "order") return -50;
                            if (d.group === "family") return -30;
                            return -30;
                        }))
                    .force("x", d3.forceX()
                        .strength(d => d.group === "family" ? 0.2 : 0.01))
                    .force("y", d3.forceY()
                        .strength(d => d.group === "family" ? 0.2 : 0.01))
                    .force("collide", d3.forceCollide()
                        .radius(d => {
                            if (d.group === "order") return 20;
                            if (d.group === "class") return 40;
                            if (d.group === "family") return 8;
                            return 15;
                        })
                        .strength(0.3))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("x", d3.forceX(width / 2).strength(0.05))
                    .force("y", d3.forceY(height / 2).strength(0.05));

                // Create links
                const link = svg.append("g")
                    .selectAll("line")
                    .data(links)
                    .join("line")
                    .attr("class", "link");

                // Create nodes
                const node = svg.append("g")
                    .selectAll("g")
                    .data(nodes)
                    .join("g")
                    .attr("class", "node")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));

                // Add circles to nodes
                node.append("circle")
                    .attr("r", d => d.radius)
                    .style("fill", d => {
                        if (d.group === "root") return "#666";
                        if (d.group === "class") return "#0066cc";  // Blue
                        if (d.group === "order") return "#00cccc";  // Cyan
                        if (d.group === "family") return "#fff";    // White
                        return color(d.status);                     // Species colors by status
                    });

                // Add hover effects to nodes
                node.on("mouseover", function(event, d) {
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);

                    let tooltipText = d.id;
                    if (d.species_name_en) {
                        tooltipText += `<br>${d.species_name_en}`;
                    }
                    if (d.group && d.group !== "species") {
                        tooltipText += ` (${d.group})`;
                    }
                    if (d.status) {
                        tooltipText += `<br>${d.status}`;
                    }

                    tooltip.html(tooltipText)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", function() {
                    tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
                });

                // Add hover effects for links
                link.on("mouseover", function(event, d) {
                    d3.select(this)
                        .style("stroke-width", "2px");
                })
                .on("mouseout", function(event, d) {
                    d3.select(this)
                        .style("stroke-width", "1px");
                });

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

                // Update on window resize
                window.addEventListener('resize', function() {
                    width = getContainerWidth();
                    simulation.force("center", d3.forceCenter(width / 2, height / 2));
                    simulation.alpha(0.3).restart();
                });
            </script>
        </div>
        """
