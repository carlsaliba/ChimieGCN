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

        # One-hot encoding for hybridization
        hybridization = atom.GetHybridization()
        hybridization_one_hot = [
            1 if hybridization == h else 0
            for h in [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
        ]
        features.extend(hybridization_one_hot)

        # Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)

        # Scalar features
        # Number of implicit hydrogens
        implicit_hydrogens = atom.GetNumImplicitHs()
        features.append(implicit_hydrogens)

        # Formal charge
        formal_charge = atom.GetFormalCharge()
        features.append(formal_charge)
        '''
        # Debug: Print extracted features
        print(f"Atom {atom.GetIdx()} ({atom.GetSymbol()}):")
        print(f"  Atomic Number One-Hot: {atomic_one_hot}")
        print(f"  Hybridization One-Hot: {hybridization_one_hot}")
        print(f"  Aromaticity: {features[-2]}")  # Implicit hydrogens
        print(f"  Formal Charge: {features[-1]}")  # Formal charge'''

        return features


    def smiles_to_graph(self):
        """
        Converts a molecule's SMILES string into a graph representation.
        
        This function creates two matrices:
        - `node_mat`: A padded node feature matrix of size `(max_atoms, node_vec_len)`.
        - `adj_mat`: An adjacency matrix of size `(max_atoms, max_atoms)`.
        """
        atoms = self.mol.GetAtoms()
        n_atoms = len(list(atoms))

        # Initialize node matrix with zeros, ensuring consistent padding to max_atoms
        node_mat = np.zeros((self.max_atoms, self.node_vec_len))

        for i, atom in enumerate(atoms):
            if i >= self.max_atoms:  # Stop if max_atoms limit is reached
                break
            features = self.get_atom_features(atom)  # Extract features for the atom

            # Ensure features fit within node_vec_len
            if len(features) > self.node_vec_len:
                print(f"Warning: Feature length {len(features)} exceeds node_vec_len {self.node_vec_len}. Truncating.")
                features = features[:self.node_vec_len]

            node_mat[i, :len(features)] = features  # Populate the feature matrix

        # Create adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)

        # Pad adjacency matrix to (max_atoms, max_atoms)
        if adj_mat.shape[0] < self.max_atoms:
            padding = self.max_atoms - adj_mat.shape[0]
            adj_mat = np.pad(
                adj_mat, pad_width=((0, padding), (0, padding)), mode="constant"
            )

        # Add self-loops to adjacency matrix (make each atom its own neighbor)
        adj_mat = adj_mat + np.eye(self.max_atoms)

        # Save the matrices
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

