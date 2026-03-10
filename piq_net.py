import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphEncoder(nn.Module):

    def __init__(self, in_features, hidden_dim):
        super().__init__()

        self.gat1 = GATConv(in_features, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim)

    def forward(self, x, edge_index):

        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))

        return x


class TemporalEncoder(nn.Module):

    def __init__(self, dim):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

    def forward(self, x):

        return self.encoder(x)


class PIQHead(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)
        )

    def forward(self, x):

        return self.net(x)


class PIQNet(nn.Module):

    def __init__(self, node_features, hidden_dim=64):

        super().__init__()

        self.graph_encoder = GraphEncoder(node_features, hidden_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim)
        self.head = PIQHead(hidden_dim)

    def forward(self, node_features, edge_index, temporal_sequence):

        node_embeddings = self.graph_encoder(node_features, edge_index)

        global_state = node_embeddings.mean(dim=0)

        temporal_out = self.temporal_encoder(temporal_sequence)

        time_state = temporal_out[:, -1, :]

        state = global_state + time_state

        piq = self.head(state)

        return piq
