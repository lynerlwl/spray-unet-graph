import numpy as np
import torch
from torch_geometric.data import Data
# import pickle

# with open('conv_layer/improved', 'rb') as f:
#     feature_map = (pickle.load(f)) 

# feature_map[8].shape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_grid_adjacency_matrix(rows, cols, connectivity=8):
    total_nodes = rows * cols
    adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

    def node_index(row, col):
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            current_node = node_index(row, col)

            # Connect to the node above (if it exists)
            if row > 0:
                adjacency_matrix[current_node][node_index(row - 1, col)] = 1

            # Connect to the node below (if it exists)
            if row < rows - 1:
                adjacency_matrix[current_node][node_index(row + 1, col)] = 1

            # Connect to the node on the left (if it exists)
            if col > 0:
                adjacency_matrix[current_node][node_index(row, col - 1)] = 1

            # Connect to the node on the right (if it exists)
            if col < cols - 1:
                adjacency_matrix[current_node][node_index(row, col + 1)] = 1
                
            # Connect to the node diagonally above-left (if it exists)
            if row > 0 and col > 0 and connectivity == 8:
                adjacency_matrix[current_node][node_index(row - 1, col - 1)] = 1

            # Connect to the node diagonally above-right (if it exists)
            if row > 0 and col < cols - 1 and connectivity == 8:
                adjacency_matrix[current_node][node_index(row - 1, col + 1)] = 1

            # Connect to the node diagonally below-left (if it exists)
            if row < rows - 1 and col > 0 and connectivity == 8:
                adjacency_matrix[current_node][node_index(row + 1, col - 1)] = 1

            # Connect to the node diagonally below-right (if it exists)
            if row < rows - 1 and col < cols - 1 and connectivity == 8:
                adjacency_matrix[current_node][node_index(row + 1, col + 1)] = 1

    return adjacency_matrix

def feature_map_to_graph(feature_map):
    # Assuming feature_map is a 4D tensor (batch_size, channels, height, width)
    batch_size, channels, height, width = feature_map.shape

    # Create node features (flatten each channel for each pixel)
    x = feature_map.view(-1, channels) # shape: (batch_size * height * width, channels)
    
    # x = feature_map.view(batch_size, channels, -1)  # shape: (batch_size, channels, height * width)
    # x = x.permute(0, 2, 1).contiguous().view(-1, channels)  # shape: (batch_size * height * width, channels)
    
    adjacency_matrix = create_grid_adjacency_matrix(height, width, connectivity=4) # row = height, col = width
    edge_index = torch.from_numpy(adjacency_matrix).nonzero().t().contiguous()

    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)

    return data.to(device=device)

# graph_data = feature_map_to_graph(feature_map[8])
# out = model(graph_data.x, graph_data.edge_index)

# Reverse the flattening of node features
# out2 = out.view(1, 1024, 38, 75) 


