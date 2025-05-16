import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from itertools import combinations
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl


def generate_random_points(center, radius, num_points, min_spacing=None):
    """
    Generates `num_points` randomly distributed within a given `radius` of `center`.
    Optionally ensures a minimum distance (`min_spacing`) between points.
    """
    points = []
    attempts = 0
    max_attempts = num_points * 10  # To prevent infinite loops

    while len(points) < num_points and attempts < max_attempts:
        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)
        # Random radius with sqrt distribution for even spacing
        r = radius * np.sqrt(np.random.uniform(0, 1))

        # Convert to Cartesian
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)

        # Check minimum spacing
        if min_spacing is None or all(np.linalg.norm(np.array([x, y]) - np.array(p)) >= min_spacing for p in points):
            points.append((x, y))
        
        attempts += 1
    
    return points

# def pull_data():

#     # Pull my spreadsheet. Determine categories based on multi-index columns in header
#     # Pull only non-trending, lowest dose, normalized to max dose (not great)
#     # Drop all columns with no significant cells, map each column to a node (sequential)
#     # return data, dictionary mapping column names to categories, and dictionary mapping
#     # columns to node numbers

#     viz_data = pd.read_excel("/Users/rebeccakrall/Data/Dashboard/Visualization.xlsx", skiprows =1 )
#     viz_header = pd.read_excel("/Users/rebeccakrall/Data/Dashboard/Visualization.xlsx", header = [0,1])
#     data_only = viz_data.iloc[:,5:-1]
#     column_cats = {k:v for v,k in viz_header.iloc[:,5:-1].columns}

#     lowest_dose = pd.DataFrame()
#     max_val = viz_data.iloc[:, -1]
#     for c in data_only.columns:
#         dat = []
#         for v in data_only[c]:
#             if type(v) == str:
#                 if 't' in v:
#                     dat.append(np.nan)
#                 else:
#                     dat.append(float(v.split(',')[0]))
#             else:
#                 dat.append(v)
            
#         dose = pd.Series(dat).astype(float)/max_val.astype(float)
#         lowest_dose[c] = dose
#     lowest_dose = lowest_dose.dropna(axis = 1, how= 'all')

#     # var_mapping = {}
#     # for ind,col in enumerate(lowest_dose.columns):
#     #     var_mapping[col] = ind
    
#     return lowest_dose, column_cats

# def pull_new_dat():

    category = pd.read_excel("/Users/rebeccakrall/Data/Dashboard/Category mapping.xlsx")
    new_df = category.loc[:,['Disease 1', 'Category']]
    cats = new_df.set_index('Disease 1').to_dict()

    assay_df = category.loc[:,['Assay', 'Disease 1']]
    assay_map = assay_df.set_index('Assay').to_dict()

    viz_data = pd.read_excel("/Users/rebeccakrall/Data/Dashboard/effect_size.xlsx", sheet_name= "Sheet3", index_col= 0)
    viz_data = viz_data.dropna(axis = 0, how = 'all')
    viz_data.index = viz_data.index.map(assay_map['Disease 1'])
    allrows = []
    for i in (viz_data.index).unique():
        cat = viz_data.iloc[viz_data.index == i, :].max()
        allrows.append(cat)
    df = pd.DataFrame(allrows)
    df  = df.set_index((viz_data.index).unique()).T

    df_binned = df.apply(lambda col: pd.cut(col, bins=[0, 0.1, 0.3, float('inf')], labels=[1, 2, 3]))
    pro_data = pd.read_excel("/Users/rebeccakrall/Data/Dashboard/dashboard_rework.xlsx", skiprows= 1, index_col = 0)
    all_dat = pd.concat([pro_data, df_binned])


    return all_dat, cats['Category']

def pull_all_dat():
    data = pd.read_csv('./Data/Final_Compound_Effect_size.csv', index_col = 0)
    with open('./Data/columns_categories.pkl', 'rb') as f:
        categories = pkl.load(f)

    with open('./Data/cats.pkl', 'rb') as f2:
        cats = pkl.load(f2)
    return data, categories, cats

def create_fixed_layout(cat_edges, G, categories):
    # Distribute categories using an nx layout
    dummy_graph = nx.Graph()
    dummy_graph.add_edges_from(cat_edges.keys())
    # dummy_graph.add_nodes_from(set(cat_edges.keys()))
    catpos = nx.arf_layout(dummy_graph)  # You can also use spring_layout

    # determine the average and minimum distance between categoires
    dists = []
    for a in list(combinations(catpos.keys(),2)):
        dists.append(math.dist(catpos[a[0]] , catpos[a[1]]))


    catsample = {}
    for cat, vertex in catpos.items():
        cat_count = sum([True for x in categories.values() if x == cat])
        max_jitter = min(dists)/3
        # sampler = qmc.PoissonDisk(d=2, radius = max_jitter/2)
        # catsample[cat] = list(np.random.normal(0, max_jitter, cat_count*2).reshape(cat_count,2) + catpos[cat])
        catsample[cat] = list(generate_random_points(catpos[cat], max_jitter, cat_count, min_spacing=0.2))

    pos= {}
    for node in G.nodes:
        
        # center = catpos[categories[node]]
        pos[node] = catsample[categories[node]].pop()

    return pos, {node: (pos[node][0] * 1000, pos[node][1] * 1000) for node in G.nodes()}  # Scale for Pyvis

def create_graphs():

    data, _, categories = pull_all_dat()

    # create a graph with all the possible nodes
    # G = nx.Graph()
    # G.add_nodes_from(data.columns)

    # create a dict of graphs for each individual compound
    compounds = {}
    edges = {}
    node_weights = {}
    for ind, row in data.iterrows():
        sig = row.dropna().index
        compounds[ind] = nx.Graph()
        compounds[ind].add_edges_from(list(combinations(sig,2)))

        node_weights[ind] = row[sig].to_dict()
        for edge in list(combinations(sig,2)):
            if edge in edges:
                edges[edge] = edges[edge] + 1
            else:
                edges[edge] = 1

    cat_edges = {}
    for e in edges.keys():
        new_key = (categories[e[0]], categories[e[1]])
        if new_key[0] != new_key[1]:
            if new_key in cat_edges:
                cat_edges[new_key] = cat_edges[new_key]+1
            else:
                cat_edges[new_key] = 1

    G = nx.Graph()
    G.add_edges_from(edges)
    # determine the graph layout based on the compiled graph    
    pos, layout = create_fixed_layout(cat_edges, G, categories)  # Precompute layout
    cat_colors, node_colors = create_color_dicts(categories)
    return G, compounds, edges, node_weights, pos, layout, cat_colors, node_colors

def create_color_dicts(categories):
    
    copts = ['skyblue', 'sandybrown', 'chocolate', 'mediumturquoise', 
             'blueviolet', 'cadetblue', 'lawngreen', 'lightsteelblue', 
             'mediumaquamarine',  'mediumpurple', 'plum','orangered']
    
    copts = [
        "#1F77B4",  # Muted Blue
        "#FF7F0E",  # Vivid Orange
        "#2CA02C",  # Strong Green
        "#D62728",  # Deep Red
        "#9467BD",  # Rich Purple
        "#8C564B",  # Warm Brown
        "#E377C2",  # Soft Pink
        "#7F7F7F",  # Medium Gray
        "#BCBD22",  # Lime Green
        "#17BECF",  # Cyan Blue
        "#FFA07A",  # Light Salmon
        "#4682B4",  # Steel Blue
        "#FFD700",  # Golden Yellow
    ]

    
    cat_colors = {cat: copts.pop() for cat in set(categories.values())}
    node_colors = {node: cat_colors[cat] for node, cat in categories.items()}

    return cat_colors, node_colors

def create_group_plot(edges, pos, node_colors, opacity = 0.6):
    G = nx.Graph()
    G.add_edges_from(edges)

    all_edges = []
    for edge, weight in edges.items():

        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            opacity = opacity*0.2,
            line=dict(width=weight, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        all_edges.append(edge_trace)


    for node in G.nodes():
   
        x, y = pos[node]
        node_trace = go.Scatter(
            x=[x], y=[y],
            mode='markers',
            opacity = opacity,
            hoverinfo='text',
            marker=dict(
                size=20,
                color=node_colors[node],
                line_width=0.2
            )
        )

    # Add hover functionality
    # node_trace.marker.color = 'blue'  # Default color
        node_trace.hoverinfo = 'text'
        node_trace.text = node
        all_edges.append(node_trace)


    return all_edges

def create_compound_plot(G, pos, node_colors, node_weights):

    edge_traces = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=1.2, color='#999'),
            opacity = 0.8,
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # all_nodes = []
    for node in G.nodes():
   
        x, y = pos[node]
        weight = node_weights[node]/2
        node_trace = go.Scatter(
            x=[x], y=[y],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size = 20 * weight,
                color=node_colors[node],
                line_width=0.5
            )
        )

    # Add hover functionality
    # node_trace.marker.color = 'blue'  # Default color
        node_trace.hoverinfo = 'text'
        node_trace.text = node
        edge_traces.append(node_trace)

    # all_p = edge_traces.extend(all_nodes)

    return edge_traces

def create_legend(color_legend, title="Legend"):
    """
    Generates a standalone legend in Plotly.
    """
    fig = go.Figure()

    for label, color in color_legend.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # No actual data points
            mode='markers',
            marker=dict(size=15, color=color),
            name=label
        ))

    fig.update_layout(
        title='',
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        width=200,
        legend=dict(
            x=0,
            xanchor='left',
            y=1, # Adjust y and yanchor as needed for vertical positioning
            yanchor='top'
        )
    )

    return fig

def create_heatmap(categories, data):
    
    reverse_categories = {}
    for k,v in categories.items():
        # print(v)
        if v in reverse_categories.keys():
            reverse_categories[v].append(k)
        else:
            reverse_categories[v] = [k]

    fig = plt.figure(figsize=(12, 6))
    df = data
    for k,v in reverse_categories.items():

        cm = sns.light_palette(cat_colors[k], as_cmap= True, reverse = True)
        df2 = df.copy()
        nan_col = [col for col in  df2.columns if not col in v]
        df2.loc[:, nan_col]= np.nan
        ax = sns.heatmap(df2, cmap = cm, square = True, linecolor='w', linewidths= 0.01, cbar = False)

    ax.collections[0].cmap.set_bad('0.95')
    ax.set_xticks([])
    ax.set_yticks([]) 

    return fig


st.set_page_config(page_title = "Melior Therapeutic Mapping" , page_icon = "Melior_Logo_small.jpg")
# st.image('Melior_Logo.jpg', width = 200)
st.title("Mapping Versatiliy Across Therapeutic Areas")

if 'G' not in st.session_state:
    G, compounds, edges, node_weights, pos, node_positions, cat_colors, node_colors = create_graphs()
    
    st.session_state['G'] = G
    st.session_state['compounds'] = compounds
    st.session_state['edges'] = edges
    st.session_state['node_weights'] = node_weights
    st.session_state['pos'] = pos
    st.session_state['node_positinos'] = node_positions
    st.session_state['cat_colors'] = cat_colors
    st.session_state['node_colors'] = node_colors
else:
    G = st.session_state['G']
    compounds = st.session_state['compounds'] 
    edges = st.session_state['edges']
    pos = st.session_state['pos']
    node_positions = st.session_state['node_positinos'] 
    cat_colors = st.session_state['cat_colors']
    node_colors = st.session_state['node_colors']
    node_weights = st.session_state['node_weights']


# Streamlit UI: allow the user to select which compound to view

# selected_compound = st.sidebar.selectbox("Select a compound:", ["All"] + list(compounds.keys()))
# perhaps rework to use a slider instead - with a 

selected_compound = color = st.select_slider(
    "Scroll through to visualize therapeutic profile of different compounds",
    options=["All"] + list(compounds.keys()),
    # value=("Common Compounds", "Test Articles")
)

# print(select_compound)
if selected_compound != 'All':
    all_plots = create_group_plot(edges, pos, node_colors, opacity = 0.3)
    edge_traces = create_compound_plot(compounds[selected_compound], pos, node_colors, node_weights[selected_compound])
    all_plots.extend(edge_traces)
    # Combine edge traces and node trace
    fig = go.Figure(data=all_plots)
else:
    all_plots = create_group_plot(edges, pos, node_colors, opacity = 1)
    fig = go.Figure(data = all_plots)


# Create layout for the graph
fig.update_layout(
    # title="Network Visualization",
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels = False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels = False),
    width = 700,
    height = 700
)


data, _, categories = pull_all_dat()
# heatmap = create_heatmap(categories, data)
leg = create_legend({k:cat_colors[k] for k in list(set([categories[x] for x in G.nodes]))})
col1, col2 = st.columns([0.9, 0.1]) 
# Display the plot in Streamlit
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.plotly_chart(leg, use_container_width=True)
    
# st.pyplot(heatmap, use_container_width = True)