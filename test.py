# Load the dataset and visualize transformations for the first molecule
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from chem_gcn.model import ChemGCN
from chem_gcn.utils import (
    train_model,
    test_model,
    parity_plot,
    loss_curve,
    Standardizer,
)
from karlchimie.graphs_karl import GraphData, collate_graph_dataset




from visualisation import visualize_molecular_graph

np.random.seed(0)
torch.manual_seed(0)
use_GPU = torch.cuda.is_available()

#### Inputs
max_atoms = 200
node_vec_len = 60
train_size = 0.7
batch_size = 32
hidden_nodes = 60
n_conv_layers = 4
n_hidden_layers = 2
learning_rate = 0.01
n_epochs = 50

#### Start by creating dataset
main_path = Path(__file__).resolve().parent
data_path = main_path / "data" / "solubility_data.csv"
dataset = GraphData(
    dataset_path=data_path, max_atoms=max_atoms, node_vec_len=node_vec_len
)

# Access the first data point
(node_mat, adj_mat,additional_props), output, smile = dataset[100]

# Print the results
#visualize_molecular_graph(smile)
print("SMILES:", smile)
print("Node Matrix (Shape: {}):\n".format(node_mat.shape), node_mat[:5,-10:])
print("Adjacency Matrix (Shape: {}):\n".format(adj_mat.shape), adj_mat)
print("Output (Measured log solubility):", output)
print("additional features:", additional_props)




