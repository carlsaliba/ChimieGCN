import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops


def visualize_molecular_graph(smiles):
    """
    Visualizes the molecular graph of a given SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Get adjacency matrix from RDKit
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)

    # Create a NetworkX graph from the adjacency matrix
    G = nx.Graph(adj_matrix)

    # Add node labels (atomic numbers) from RDKit molecule
    node_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Visualize the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(
        G, pos, with_labels=True, labels=node_labels, node_color="lightblue", font_weight="bold"
    )
    plt.title(f"Molecular Graph for SMILES: {smiles}")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example SMILES strings
    smiles_examples = ["CCO", "C1=CC=CC=C1", "C(C(=O)O)N"]  # Ethanol, Benzene, Alanine

    for smiles in smiles_examples:
        visualize_molecular_graph(smiles)
