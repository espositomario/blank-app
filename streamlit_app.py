import streamlit as st
import plotly.express as px
import pandas as pd
import pickle

# Load data
CODINGS_SIZE = 6
INPUT_GENES = 'ALL'
INPUT_FEATURES = 'X_FC'
INPUT_NORM = 'z'
METHOD = 'VAE'
k = 80
LABELS_COL = f'GMM_{METHOD}_{k}'
ID = f'{CODINGS_SIZE}D_{INPUT_GENES}_{INPUT_FEATURES}_{INPUT_NORM}'
DIR_DATA = f'./data/{ID}_analysis/{LABELS_COL}/'

# Load gene cluster dictionary
with open(f'./gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load CODE and LOG matrices
CODE = pd.read_csv(f'./CODE.csv', index_col='GENE')
LOG = pd.read_csv(f'./ALL_X_FC.csv').set_index('GENE')

# Map cluster IDs to CODE and LOG
gene_to_cluster = {}
for cluster_id, gene_list in GENE_CLUSTERS.items():
    for gene in gene_list['gene_list']:
        gene_to_cluster[gene] = cluster_id

LOG[LABELS_COL] = LOG.index.map(gene_to_cluster).astype(int)
CODE[LABELS_COL] = LOG.index.map(gene_to_cluster).astype(int)

# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]

# Streamlit app layout
st.set_page_config(layout="wide")  # Enable wide layout

st.title("Gene Visualization in VAE-UMAP Space")

# Dropdown for feature selection
selected_feature = st.selectbox(
    "Select Continuous Feature to Color By",
    options=continuous_features,
    index=0  # Default to the first feature
)

# Filter data using color scale
color_min, color_max = st.slider(
    f"Select Range on {selected_feature} Color Scale",
    min_value=float(CODE[selected_feature].min()),
    max_value=float(CODE[selected_feature].max()),
    value=(float(CODE[selected_feature].min()), float(CODE[selected_feature].max())),
    step=0.01
)
filtered_data = CODE[(CODE[selected_feature] >= color_min) & (CODE[selected_feature] <= color_max)]

# Layout for scatter plot and violin plot
col1, col2 = st.columns(2)

# UMAP Scatter plot settings
with col1:
    st.header("UMAP Scatter Plot")
    # Compact settings dropdown
    with st.expander("⚙️", expanded=False):
        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                font-size: 12px !important;
                padding: 5px 5px !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        point_size = st.slider("Point Size", min_value=1, max_value=20, value=5, step=1)
        colormap = st.selectbox("Select Colormap", ["viridis", "plasma", "cividis", "rainbow", "magma"], index=0)

    # Create scatter plot
    fig = px.scatter(
        filtered_data,
        x="UMAP1",
        y="UMAP2",
        color=selected_feature,
        hover_data=["UMAP1", "UMAP2", selected_feature],
        title=f"2D Scatter Plot Colored by {selected_feature}",
        labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
        color_continuous_scale=colormap
    )

    # Adjust plot appearance
    fig.update_traces(marker=dict(size=point_size))  # Adjust point size
    fig.update_layout(
        xaxis=dict(scaleanchor="y"),  # Enforce 1:1 aspect ratio
        yaxis=dict(scaleanchor="x"),
        xaxis_showgrid=False,  # No gridlines
        yaxis_showgrid=False,
        xaxis_tickvals=[],  # Hide tick labels
        yaxis_tickvals=[],
        plot_bgcolor="white",
        autosize=False,
        width=500,  # Fixed width
        height=500  # Fixed height for a square layout
    )
    st.plotly_chart(fig, use_container_width=True)

# Violin plot settings
with col2:
    st.header("Distribution Plot")
    # Compact settings dropdown
    with st.expander("⚙️", expanded=False):
        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                font-size: 12px !important;
                padding: 5px 5px !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        show_outliers = st.checkbox("Show Outliers in Box Plot", value=False)

    # Create violin plot
    box_violin_fig = px.violin(
        CODE,
        y=selected_feature,
        box=True,  # Add a boxplot inside
        points=False,  # Disable strip plot
        title=f"Distribution of {selected_feature}",
        color_discrete_sequence=["#B0B0B0"]  # Gray color
    )

    # Toggle outliers in the box plot
    if not show_outliers:
        box_violin_fig.update_traces(box_visible=True, meanline_visible=False)

    # Adjust plot appearance
    box_violin_fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showticklabels=False),  # Hide x-axis ticks
    )
    st.plotly_chart(box_violin_fig, use_container_width=True)
