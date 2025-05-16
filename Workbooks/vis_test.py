# import streamlit as st
# import networkx as nx
# from pyvis.network import Network
# import tempfile
# import os
# import pandas as pd
# from itertools import combinations

# # Create a fixed layout for all graphs
# def create_fixed_layout(G):
#     pos = nx.shell_layout(G)  # You can also use spring_layout
#     return {node: (pos[node][0] * 1000, pos[node][1] * 1000) for node in G.nodes()}  # Scale for Pyvis

# # Generate the graph and layout
def create_graph():
    G = nx.Graph()
    # compounds = {
    #     "A": [(1, 2), (2, 3), (3, 1)],
    #     "B": [(2, 3), (3, 4), (4, 2)],
    #     "C": [(3, 5), (5, 6), (6, 3)],
    #     "D": [(7, 8), (8, 9), (9, 7)]
    # }
    viz_data = pd.read_csv("/Users/rebeccakrall/Desktop/Visualization.csv", skiprows =1 )
    var_mapping = {}
    for ind,col in enumerate(viz_data.iloc[:,5:-1].columns):
        var_mapping[col] = ind

    assays = viz_data.iloc[:, 5:-1].columns
    compounds = {}
    for ind, row in viz_data.iloc[:,5:-1].iterrows():
        exists =  ~row.isna()
        trend = row.notna() & row.str.contains('t')
        sig = assays[exists & ~trend]
        
        compounds[str(ind)] = []
        for edge in list(combinations(sig,2)):
            if var_mapping[edge[0]] not in G.nodes():
                G.add_node(var_mapping[edge[0]])
            if var_mapping[edge[1]] not in G.nodes():
                G.add_node(var_mapping[edge[1]])
            

            # print(f'Edges: {var_mapping[edge[0]]}, {var_mapping[edge[1]]}')
            G.add_edge(var_mapping[edge[0]], var_mapping[edge[1]])
            compounds[str(ind)].append((var_mapping[edge[0]], var_mapping[edge[1]]))
    
    layout = create_fixed_layout(G)  # Precompute layout
    return G, compounds, layout

# # Streamlit UI
# st.sidebar.title("Graph Controls")

# # Generate graph data
# G, compounds, layout = create_graph()

# selected_compound = st.sidebar.selectbox("Select a compound:", ["All"] + list(compounds.keys()))
# show_labels = st.sidebar.checkbox("Show node labels", False)



# # Create Pyvis Network
# net = Network(height="600px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
# net.repulsion()
# net.from_nx(G)

# # Apply consistent layout
# for node, pos in layout.items():
#     net.get_node(node)["x"] = pos[0]
#     net.get_node(node)["y"] = pos[1]

# # Highlight selected compound
# if selected_compound != "All":
#     highlighted_edges = compounds[selected_compound]
#     for edge in G.edges():
#         if edge in highlighted_edges or tuple(reversed(edge)) in highlighted_edges:
#             net.get_node(edge[0])["color"] = "red"
#             net.get_node(edge[1])["color"] = "red"
#         else:
#             net.get_node(edge[0])["color"] = "gray"
#             net.get_node(edge[1])["color"] = "gray"

# # Show labels if checkbox is selected
# if show_labels:
#     for node in net.nodes:
#         node["label"] = str(node["id"])  # Add labels

# # Save graph to temporary HTML file
# with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
#     tmp_path = tmp_file.name
#     net.save_graph(tmp_path)

# # Embed HTML in Streamlit
# st.components.v1.html(open(tmp_path, "r").read(), height=600)

# # Clean up temp file
# os.unlink(tmp_path)

# import plotly.graph_objects as go
# import networkx as nx
# import streamlit as st

# # Create a simple graph using NetworkX
# G = nx.karate_club_graph()

# # Create position dictionary using spring layout
# pos = nx.spring_layout(G)

# # Create Plotly graph
# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x.append(x0)
#     edge_x.append(x1)
#     edge_y.append(y0)
#     edge_y.append(y1)

# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines'
# )

# node_x = []
# node_y = []
# for node in G.nodes():
#     x, y = pos[node]
#     node_x.append(x)
#     node_y.append(y)

# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         colorscale='YlGnBu',
#         size=10,
#         colorbar=dict(thickness=15, title="Node Connections", xanchor="left")
#     )
# )

# # Add hover information
# node_trace.marker.color = [len(list(G.neighbors(node))) for node in G.nodes()]
# node_trace.marker.size = [len(list(G.neighbors(node))) * 2 for node in G.nodes()]

# # Create the layout for the graph
# layout = go.Layout(
#     title="Karate Club Network",
#     showlegend=False,
#     hovermode='closest',
#     xaxis=dict(showgrid=False, zeroline=False),
#     yaxis=dict(showgrid=False, zeroline=False)
# )

# # Create the figure
# fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

# # # Display the figure in Streamlit
# st.plotly_chart(fig)


### MOST PROMISING
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a simple graph using NetworkX
G = nx.karate_club_graph()

# Create position dictionary using spring layout
pos = nx.spring_layout(G)
node_positions = {node: (x * 1000, y * 1000) for node, (x, y) in pos.items()}

# Example compounds with their edges
compounds = {
    "Compound A": [(0, 1), (1, 2), (2, 3)],
    "Compound B": [(2, 3), (3, 4), (4, 5)],
    "Compound C": [(5, 6), (6, 7), (7, 8)]
}

# Streamlit UI: allow the user to select which compound to view
selected_compound = st.sidebar.selectbox("Select a compound:", ["All"] + list(compounds.keys()))

# Create Plotly graph
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_y.append(y0)
    edge_y.append(y1)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        size=10,
        color='blue',
        line_width=2
    )
)

# Add hover functionality
node_trace.marker.color = 'blue'  # Default color
node_trace.hoverinfo = 'text'
node_trace.text = [f'Node {node}' for node in G.nodes()]

# Filter edges for the selected compound
edges_to_display = []
if selected_compound != "All":
    compound_edges = compounds[selected_compound]
    for edge in compound_edges:
        if edge in G.edges() or (edge[1], edge[0]) in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edges_to_display.append([(x0, y0), (x1, y1)])

# Create edge traces for the selected compound
edge_trace_selected = go.Scatter(
    x=[x for x, y in edges_to_display], 
    y=[y for x, y in edges_to_display],
    line=dict(width=2, color='red'),
    hoverinfo='none',
    mode='lines'
)

# Combine edge traces and node trace
fig = go.Figure(data=[edge_trace, edge_trace_selected, node_trace])

# Create layout for the graph
fig.update_layout(
    title="Network Visualization",
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)

# Display the plot in Streamlit
st.plotly_chart(fig)
