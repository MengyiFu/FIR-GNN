import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, TAGConv

from models.layer import NLSAttention


class FIRGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim, heads=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden)
        self.norm1 = BatchNorm(hidden)
        self.conv2 = GATConv(hidden, hidden // 2, heads, concat=False)
        self.norm2 = BatchNorm(hidden // 2)
        self.conv3 = GCNConv(hidden // 2, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.norm1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.norm2(x)
        x = self.conv3(x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GAT
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim, heads=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(input_dim, hidden, heads, concat=False)
        self.conv2 = GATConv(hidden, output_dim, heads, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GCN_GAT
class GCNGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim, heads=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GATConv(hidden, hidden // 2, heads, concat=False)
        self.conv3 = GCNConv(hidden // 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class FTGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim, heads=2):
        super().__init__()
        self.attention = NLSAttention(input_dim)
        self.conv1 = TAGConv(input_dim, hidden, K=heads)
        self.conv2 = TAGConv(hidden, output_dim, K=heads)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.attention(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GNNFactory:
    @staticmethod
    def create_model(config, input_dim, output_dim):
        model_name = config["model"]["name"]
        common_params = {
            "input_dim": input_dim,
            "hidden": config["model"]["hidden"],
            "output_dim": output_dim,
            "heads": config["model"]["heads"]
        }

        if model_name == "FIR-GNN":
            return FIRGNN(**{**common_params, "heads": config["model"]["heads"]})
        elif model_name == "GCN":
            return GCN(**common_params)
        elif model_name == "GAT":
            return GAT(**{**common_params, "heads": config["model"]["heads"]})
        elif model_name == "GCN_GAT":
            return GCNGAT(**{**common_params, "heads": config["model"]["heads"]})
        else:
            return FTGCN(**{**common_params, "heads": config["model"]["heads"]})