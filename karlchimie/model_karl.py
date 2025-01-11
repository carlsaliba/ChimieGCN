import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    def __init__(self, node_in_len: int, node_out_len: int):
        super().__init__()
        self.conv_linear = nn.Linear(node_in_len, node_out_len)
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)
        idx_mat = torch.eye(adj_mat.shape[-2], adj_mat.shape[-1], device=n_neighbors.device)
        idx_mat = idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)
        node_fea = self.conv_linear(node_fea)
        node_fea = self.conv_activation(node_fea)

        return node_fea


class PoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_fea):
        return node_fea.mean(dim=1)


class ChemGCN(nn.Module):
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        n_additional_props: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        self.conv_layers = nn.ModuleList(
            [
                ConvolutionLayer(
                    node_in_len=node_fea_len,
                    node_out_len=node_fea_len,
                )
                for _ in range(n_conv)
            ]
        )

        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        self.pooling_activation = nn.LeakyReLU()

        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        self.additional_props_layer = nn.Linear(
            hidden_fea_len + n_additional_props, hidden_fea_len
        )

        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)
        self.hidden_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)

        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [self.hidden_layer for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
            )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat, additional_props):
        node_fea = self.init_transform(node_mat)

        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)

        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)

        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        combined_features = torch.cat([hidden_node_fea, additional_props], dim=1)
        hidden_node_fea = self.additional_props_layer(combined_features)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)

        out = self.hidden_to_output(hidden_node_fea)

        return out

if __name__ == "__main__":
    from graphs_karl import Graph

    g = Graph("CC", node_vec_len=20)
    n = torch.Tensor(g.node_mat).view(1, g.node_mat.shape[0], g.node_mat.shape[1])
    a = torch.Tensor(g.adj_mat).view(1, g.adj_mat.shape[0], g.adj_mat.shape[1])
    additional_props = torch.Tensor([[0.5, 1, 200, 1, 2, 3, 50]])  # Example additional properties

    model = ChemGCN(
        node_vec_len=20,
        node_fea_len=20,
        hidden_fea_len=10,
        n_conv=2,
        n_hidden=2,
        n_outputs=1,
        n_additional_props=6,
    )

    with torch.no_grad():
        out = model(n, a, additional_props)
        print(out)
