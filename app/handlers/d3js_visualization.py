"""Module for creating and displaying D3.js-based species visualization in Streamlit."""

import streamlit as st
try:  
    import streamlit.components.v1 as components
except ImportError:
    components = None

def create_circle_packing_html(data, width=700, height=600):
    """Create the HTML/JavaScript code for D3.js visualization."""
    return """
    <div id="visualization" style="width: """ + str(width) + """px; height: """ + str(height) + """px;">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            #visualization {
                position: relative;
                overflow: hidden;
            }
            .node {
                stroke: #000;
                stroke-width: 1px;
            }
            .node:hover {
                stroke-width: 1px;
            }
            .label {
                font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;
                text-anchor: middle;
                fill: white;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.8), -2px -2px 4px rgba(0,0,0,0.8),
                            2px -2px 4px rgba(0,0,0,0.8), -2px 2px 4px rgba(0,0,0,0.8);
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
            // Set up the D3.js visualization
            const width = """ + str(width) + """;
            const height = """ + str(height) + """;
            const margin = {top: 10, right: 10, bottom: 10, left: 10};
            
            // Create tooltip div
            const tooltip = d3.select("#visualization")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);
            
            const svg = d3.select("#visualization")
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);

            const data = {data_placeholder};

            // Create a hierarchical data structure
            const root = d3.hierarchy(data)
                .sum(d => d.value)
                .sort((a, b) => b.value - a.value);

            // Create a pack layout
            const pack = d3.pack()
                .size([width - margin.left - margin.right, height - margin.top - margin.bottom])
                .padding(3);

            // Apply the pack layout to the data
            pack(root);

            const color = d3.scaleOrdinal()
                .domain(['Extinct', 'Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened', 'Least Concern', 'Data Deficient'])
                .range([
                    '#67000d',  // Extinct - Very Dark Red
                    '#d73027',  // Critically Endangered - Red
                    '#fc8d59',  // Endangered - Orange
                    '#fee08b',  // Vulnerable - Yellow
                    '#d9ef8b',  // Near Threatened - Light Green
                    '#91cf60',  // Least Concern - Green
                    '#808080'   // Data Deficient - Gray
                ]);

            const node = svg.selectAll('.node')
                .data(root.descendants().slice(1))
                .join('circle')
                .attr('class', 'node')
                .attr('fill', d => d.children ? '#fff' : color(d.data.status))
                .attr('pointer-events', 'all')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', d => d.r);

            const label = svg.append('g')
                .attr('pointer-events', 'none')
                .attr('text-anchor', 'middle')
                .selectAll('text')
                .data(root.descendants())
                .join('text')
                .style('fill-opacity', d => d.parent === root ? 1 : 0)
                .style('display', d => d.parent === root ? 'inline' : 'none')
                .text(d => d.data.name);

            // Add interactivity with tooltip
            node.on('mouseover', function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                    
                let tooltipText = d.data.name;
                if (!d.children && d.data.status) {  // If it's a leaf node (species)
                    tooltipText += ` (${d.data.status})`;
                }
                
                tooltip.html(tooltipText)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
                    
                // Update label visibility
                label.style('display', n => n.parent === d ? 'inline' : 'none')
                    .style('fill-opacity', n => n.parent === d ? 1 : 0);
                
                d3.select(this)
                    .style("stroke", "#000")
                    .style("stroke-width", "1.5px");
            });

            node.on('mouseout', function() {
                const d = d3.select(this).datum();
                label.style('display', n => n.parent === root ? 'inline' : 'none')
                    .style('fill-opacity', n => n.parent === root ? 1 : 0);
            });
        </script>
    </div>
    """

def display_species_visualization(data):
    """Display the species visualization in Streamlit."""
    # pylint: disable=no-member
    col1, col2 = st.columns([3, 1])  # 3:1 ratio for visualization:description

    with col1:
        # Create the HTML with the data embedded
        html_content = create_circle_packing_html(data, width=700, height=600)
        html_content = html_content.replace('{data_placeholder}', str(data))
        # Display the visualization using Streamlit components
        components.html(html_content, height=600, width=700)

    with col2:
        st.markdown("### Species Hierarchy")
        st.markdown("""
        This visualization shows the hierarchical relationship of endangered species:

        **Structure:**
        - Outer circles represent families
        - Inner circles represent individual species
        """)
        
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
        - Hover over circles to see details
        - Circle size indicates relative grouping
        """)

def create_tree_html(data, width=950, height=800):
    """Create the HTML/JavaScript code for D3.js tree visualization."""
    return """
    <div id="tree-container" style="width: 100%;">
        <div id="tree-visualization" style="width: 100%; height: """ + str(height) + """px;">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
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
                    stroke: #999;
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
                const margin = {top: 20, right: 40, bottom: 30, left: 2};
                
                function getWidth() {
                    return document.getElementById('tree-visualization').offsetWidth;
                }

                // Create tooltip
                const tooltip = d3.select("#tree-visualization")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);

                // Create SVG with ID
                const svg = d3.select("#tree-visualization")
                    .append("svg")
                    .attr("id", "tree-svg")
                    .append("g")
                    .attr("transform", `translate(${margin.left},${margin.top})`);

                const data = {data_placeholder};

                // Color scale definition
                const color = d3.scaleOrdinal()
                    .domain(['Extinct', 'Critically Endangered', 'Endangered', 'Vulnerable', 
                            'Near Threatened', 'Least Concern', 'Data Deficient'])
                    .range(['#67000d', '#d73027', '#fc8d59', '#fee08b', 
                           '#d9ef8b', '#91cf60', '#808080']);

                function updateVisualization() {
                    const width = getWidth();
                    const innerWidth = width - margin.left - margin.right;
                    const innerHeight = height - margin.top - margin.bottom;

                    // Update SVG dimensions
                    d3.select("#tree-svg")
                        .attr("width", width)
                        .attr("height", height)
                        .attr("viewBox", `0 0 ${width} ${height}`);

                    // Create tree layout
                    const treeLayout = d3.tree()
                        .size([innerHeight, innerWidth]);

                    // Generate tree data
                    const root = d3.hierarchy(data);
                    const treeData = treeLayout(root);

                    // Adjust node positions
                    treeData.descendants().forEach(d => {
                        d.y = d.depth * (innerWidth / 5);
                    });

                    // Update links
                    const link = svg.selectAll(".link")
                        .data(treeData.links().filter(d => d.source.depth > 0));

                    link.exit().remove();

                    link.enter()
                        .append("path")
                        .attr("class", "link")
                        .merge(link)
                        .transition()
                        .duration(750)
                        .attr("d", d3.linkHorizontal()
                            .x(d => d.y)
                            .y(d => d.x));

                    // Update nodes
                    const node = svg.selectAll(".node")
                        .data(treeData.descendants().slice(1));

                    node.exit().remove();

                    const nodeEnter = node.enter()
                        .append("g")
                        .attr("class", "node");

                    nodeEnter.append("circle");
                    nodeEnter.append("text");

                    const allNodes = nodeEnter.merge(node);

                    allNodes.transition()
                        .duration(750)
                        .attr("transform", d => `translate(${d.y},${d.x})`);

                    // Add circles to nodes
                    allNodes.select("circle")
                        .attr("r", d => {
                            if (d.depth === 1) return 10;
                            if (d.depth === 2) return 8;
                            if (d.depth === 3) return 6;
                            return 4;
                        })
                        .style("fill", d => {
                            if (d.depth === 1) return "#0066cc";      // Class (first level) - Blue
                            if (d.depth === 2) return "#00cccc";      // Order (second level) - Cyan
                            if (d.depth === 3) return "#fff";         // Family (third level) - White
                            return color(d.data.status);              // Species (leaf nodes) - Status color
                        })
                        .style("stroke", "#000")
                        .style("stroke-width", "1px");

                    // Add hover effects
                    allNodes.on("mouseover", function(event, d) {
                        // Highlight node
                        d3.select(this).select("circle")
                            .style("stroke-width", "2px");

                        // Show tooltip
                        let tooltipText = d.data.name;
                        if (d.data.status) {
                            tooltipText += ` (${d.data.status})`;
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

                    // Add labels
                    allNodes.select("text")
                        .filter(d => d.children)
                        .attr("dy", ".31em")
                        .attr("x", d => d.children ? -10 : 10)
                        .style("text-anchor", d => d.children ? "end" : "start")
                        .style("font-size", "10px")
                        .text(d => d.data.name);
                }

                // Update visualization on window resize
                window.addEventListener('resize', updateVisualization);

                // Initialize visualization
                updateVisualization();
            </script>
        </div>
    </div>
    """

def display_tree(data):
    """Display the tree visualization in Streamlit."""
    # pylint: disable=no-member
    st.markdown("""
        <style>
            .element-container {
                width: 100%;
            }
            iframe {
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])  # 3:1 ratio for visualization:description

    with col1:
        # Create the HTML with the data embedded
        html_content = create_tree_html(data, width=950, height=1200)
        html_content = html_content.replace('{data_placeholder}', str(data))
        # Display the visualization using Streamlit components
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

def create_force_html(data, width=950, height=800):
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
                            nodes.push({ 
                                id: speciesId,
                                displayName: species.name,
                                group: "species",
                                status: species.status,
                                radius: 4 
                            });
                            links.push({ 
                                source: family.name, 
                                target: speciesId 
                            });
                        });
                    });
                });
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
                        if (d.source.group === "class") return 30;
                        if (d.source.group === "order") return 3;
                        if (d.source.group === "family") return 20;
                        return 15;
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
    
                let tooltipText = d.id;  // Changed from d.name to d.id
                if (d.group && d.group !== "species") {
                    tooltipText += ` (${d.group})`;  // Add the group type (class/order/family)
                }
                if (d.status) {
                    tooltipText += ` (${d.status})`;  // Add status if it exists
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

def display_force_visualization(data):
    """Display the force-directed visualization in Streamlit."""
    # Add responsive CSS
    # pylint: disable=no-member
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
        html_content = create_force_html(data)
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
