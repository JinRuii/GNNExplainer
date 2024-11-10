import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch_geometric.nn import GATConv, SAGEConv, GNNExplainer  # In version 2.0.4, import GNNExplainer from torch_geometric.nn
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

# Define the HybridModel class (should match the model structure used during training)
class HybridModel(torch.nn.Module):
    def __init__(self, num_features, sage_hidden_units1, gat_hidden_units, sage_hidden_units2, num_heads, num_classes, dropout):
        super(HybridModel, self).__init__()
        # Define the concat variable
        self.concat = True  # Or False, depending on model configuration

        # First layer: GraphSAGE
        self.sage_conv1 = SAGEConv(num_features, sage_hidden_units1)
        # Second layer: GAT
        self.gat_conv = GATConv(
            sage_hidden_units1,
            gat_hidden_units,
            heads=num_heads,
            dropout=dropout,
            concat=self.concat
        )
        # Determine GAT layer output dimension based on concat
        gat_output_dim = gat_hidden_units * num_heads if self.concat else gat_hidden_units

        # Third layer: GraphSAGE
        self.sage_conv2 = SAGEConv(gat_output_dim, sage_hidden_units2)
        # Classifier
        self.classifier = torch.nn.Linear(sage_hidden_units2, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First layer: GraphSAGE
        x = self.sage_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer: GAT
        x = self.gat_conv(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third layer: GraphSAGE
        x = self.sage_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier
        output = self.classifier(x)
        return output

# Set up device
device = torch.device('cpu')  # Use 'cuda' if GPU is available

# Define hyperparameters (should match those used during training)
hyperparams = {
    'epochs': 500,
    'learning_rate': 0.001,
    'dropout': 0.5,
    'sage_hidden_units1': 64,
    'gat_hidden_units': 64,
    'sage_hidden_units2': 64,
    'num_heads': 2,
}

# Label mapping
label_mapping = {'none': 0, 'low': 1, 'relatively low': 2, 'relatively high': 3, 'high': 4}

# 1. Load the model
output_dir = "./model_output"  # Set to actual output directory path
model_save_path = os.path.join(output_dir, 'trained_model.pth')
num_classes = len(label_mapping)
num_features = 20  # Number of features (based on data, should be 20)

# Create a model instance
model = HybridModel(
    num_features=num_features,
    sage_hidden_units1=hyperparams['sage_hidden_units1'],
    gat_hidden_units=hyperparams['gat_hidden_units'],
    sage_hidden_units2=hyperparams['sage_hidden_units2'],
    num_heads=hyperparams['num_heads'],
    num_classes=num_classes,
    dropout=hyperparams['dropout'],
).to(device)

# Load model parameters (set strict=False)
model.load_state_dict(torch.load(model_save_path, map_location=device), strict=False)
model.eval()  # Set to evaluation mode

# 2. Load data
# Set input file directory
input_dir = "./data_input"  # Set to actual input directory path

# Load feature file
variables_file = os.path.join(input_dir, 'H8-16_variables.csv')
variables_df = pd.read_csv(variables_file)

# Load edge list file
edges_file = os.path.join(input_dir, 'H8-16_edges.csv')
edges_df = pd.read_csv(edges_file)

# Extract features and normalize
feature_names = [
    'HOP',  # Housing price, measured in yuan per square meter
    'POD',  # Population density, measured in people per square meter
    'DIS_BUS',  # Distance to the nearest bus station (poi - point of interest), measured in meters
    'DIS_MTR',  # Distance to the nearest metro station (poi), measured in meters
    'POI_COM',  # Number of company POIs (points of interest) within the area
    'POI_SHOP',  # Number of shopping POIs within the area
    'POI_SCE',  # Number of scenic spot POIs within the area
    'POI_EDU',  # Number of educational POIs within the area
    'POI_MED',  # Number of medical POIs within the area
    'PR',  # Plot ratio (building area * number of floors (height/3.5m) / area)
    'OPEN',  # Sky openness ratio from street view images; if value is -999, street view data is not available
    'CAR',  # Car presence ratio in street view images
    'GREN',  # Green view index (greenness) in street view images
    'ENCL',  # Enclosure rate in street view images
    'WAL',  # Walkability index in street view images
    'IMA',  # Imageability index in street view images
    'COMP',  # Complexity or diversity in street view images
    'PM2_5',  # Concentration of PM2.5 (particulate matter), measured in μg/m³ per hour per day
    'PM10',  # Concentration of PM10 (particulate matter), measured in μg/m³ per hour per day
    'CO'  # Carbon monoxide concentration, measured in μg/m³ per hour per day
]

features = variables_df[feature_names].values
features = torch.FloatTensor(features)  # Convert features to Tensor

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
features = torch.FloatTensor(scaler.fit_transform(features.numpy()))  # Normalize using MinMaxScaler

# Load labels and map to integers
labels = variables_df['TD_label'].map(label_mapping).values
labels = torch.LongTensor(labels)  # Convert labels to Tensor

# Construct edge index
edge_index = torch.tensor([edges_df['_OID_1'].values, edges_df['_OID_2'].values], dtype=torch.long)
# If edges are undirected, add reverse edges
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
# Create data object
data = Data(x=features.to(device), edge_index=edge_index.to(device))

# 3. Calculate node and edge contributions using GNNExplainer
from torch_geometric.nn import GNNExplainer  # Correctly import GNNExplainer

# Initialize GNNExplainer
explainer = GNNExplainer(model, epochs=500)  # In version 2.0.4, no need to specify return_type

# Create output directory to save results
result_dir = "./result_output"  # Set to actual result output directory path
os.makedirs(result_dir, exist_ok=True)

# Save node feature importance
node_feat_importance = []
# Save edge importance
edge_importance = []

# Loop through each node for explanation
for node_idx in range(data.num_nodes):
    # Get the true label of the node
    true_label = labels[node_idx].item()
    # Use GNNExplainer to explain the node
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    # edge_mask corresponds to edge importance related to data.edge_index

    # Save node feature importance
    node_feat_importance.append([node_idx] + node_feat_mask.cpu().detach().numpy().tolist())

    # Get indices of edges related to this node
    related_edge_indices = torch.nonzero((data.edge_index[0] == node_idx) | (data.edge_index[1] == node_idx)).squeeze()
    related_edges = data.edge_index[:, related_edge_indices]

    # Retrieve importance of corresponding edges
    related_edge_mask = edge_mask[related_edge_indices.cpu().numpy()]

    # Save edge importance
    for i in range(related_edges.size(1)):
        src = related_edges[0, i].item()
        dst = related_edges[1, i].item()
        importance = related_edge_mask[i].item()
        edge_importance.append({'src': src, 'dst': dst, 'importance': importance})

    print(f"Explanation for node {node_idx} completed.")

# 4. Save node and edge contribution results
# Save node feature importance
node_feat_imp_df = pd.DataFrame(node_feat_importance, columns=['node_idx'] + feature_names)
node_feat_imp_df.to_csv(os.path.join(result_dir, 'node_feature_importance.csv'), index=False)

# Save edge importance
edge_imp_df = pd.DataFrame(edge_importance)
edge_imp_df.to_csv(os.path.join(result_dir, 'edge_importance.csv'), index=False)

print("Node and edge contributions calculated and saved to CSV files.")
