import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset

class Graph:
    def __init__(self, molecule_smiles: str, node_vec_len: int, max_atoms: int = None):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()

    def smiles_to_mol(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return
        self.mol = Chem.AddHs(mol)

    def get_atom_features(self, atom):
        features = []

        # One-hot encoding for atomic number
        atomic_number = atom.GetAtomicNum()
        atomic_one_hot = [1 if i == atomic_number else 0 for i in range(self.node_vec_len)]
        features.extend(atomic_one_hot)

        return features

    def smiles_to_graph(self):
        atoms = self.mol.GetAtoms()
        n_atoms = len(list(atoms)) if self.max_atoms is None else self.max_atoms
        
        # Dynamically calculate the feature size
        sample_atom = atoms[0]
        feature_size = len(self.get_atom_features(sample_atom))
        
        # Initialize node matrix with the correct feature size
        node_mat = np.zeros((n_atoms, feature_size))

        for atom in atoms:
            atom_index = atom.GetIdx()
            features = self.get_atom_features(atom)
            node_mat[atom_index, :len(features)] = features

        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")
        adj_mat = adj_mat + np.eye(n_atoms)

        self.node_mat = node_mat
        self.adj_mat = adj_mat


class GraphData(Dataset):
    def __init__(self, dataset_path: str, node_vec_len: int, max_atoms: int):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        df = pd.read_csv(dataset_path)
        self.indices = df.index.to_list()
        self.smiles = df["smiles"].to_list()
        self.outputs = df["measured log solubility in mols per litre"].to_list()
        self.additional_properties = df[[
            "Minimum Degree",
            "Molecular Weight",
            "Number of H-Bond Donors",
            "Number of Rings",
            "Number of Rotatable Bonds",
            "Polar Surface Area",
        ]].values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        smile = self.smiles[i]
        mol = Graph(smile, self.node_vec_len, self.max_atoms)
        node_mat = torch.Tensor(mol.node_mat)
        adj_mat = torch.Tensor(mol.adj_mat)
        additional_props = torch.Tensor(self.additional_properties[i])
        output = torch.Tensor([self.outputs[i]])
        return (node_mat, adj_mat, additional_props), output, smile


def collate_graph_dataset(batch):
    node_mats = [b[0][0] for b in batch]
    adj_mats = [b[0][1] for b in batch]
    additional_props = [b[0][2] for b in batch]
    outputs = [b[1] for b in batch]

    return (
        (torch.stack(node_mats, dim=0), torch.stack(adj_mats, dim=0), torch.stack(additional_props, dim=0)),
        torch.stack(outputs, dim=0),
    )
