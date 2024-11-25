import streamlit as st
import pandas as pd
import plotly.express as px
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

# Streamlit app
st.title("Gene Visualization in VAE-UMAP Space")

# Dropdown for feature selection
selected_feature = st.selectbox(
    "Select Feature to Color By:",
    options=continuous_features + [LABELS_COL],
    index=len(continuous_features)  # Default is the last option (Cluster Labels)
)

# Generate scatter plot
if selected_feature in CODE.columns:
    # Determine if selected feature is categorical or continuous
    if selected_feature == LABELS_COL:
        # Treat cluster labels as a categorical feature
        fig = px.scatter(
            CODE,
            x="UMAP1",
            y="UMAP2",
            color=CODE[selected_feature].astype(str),  # Cast to string for categorical coloring
            hover_data=["UMAP1", "UMAP2", selected_feature],
            title=f"2D Scatter Plot Colored by {selected_feature} (Categorical)",
            labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"}
        )
    else:
        # Treat as a continuous feature
        fig = px.scatter(
            CODE,
            x="UMAP1",
            y="UMAP2",
            color=selected_feature,
            hover_data=["UMAP1", "UMAP2", selected_feature],
            title=f"2D Scatter Plot Colored by {selected_feature} (Continuous)",
            labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"}
        )

    # Display scatter plot
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Selected feature is not available in the data!")
