# %%
from torch.utils.tensorboard import SummaryWriter
import optuna
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv
import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import HeteroData

# %%
X = pl.read_parquet("../../data/Processed_and_split/Processed_X.parquet")
y = pl.read_parquet("../../data/Processed_and_split/Processed_y.parquet")

# %%
src_ips = X['Source IP'].unique()
dest_ips = X['Destination IP'].unique()
unique_hosts = list(set(src_ips) | set(dest_ips))
host_id_map = {v: k for k, v in enumerate(unique_hosts)}
num_hosts = len(unique_hosts)
num_hosts

# %%
host_features = torch.zeros(num_hosts, 0)

# %%
flow_features_df = X.drop(['Source IP', 'Destination IP', 'Source Port', 'Destination Port'])
flow_features = torch.tensor(flow_features_df.to_numpy(), dtype=torch.float)
num_flows = flow_features.shape[0]

# %%
src_ids = X["Source IP"].to_list()
dst_ids = X["Destination IP"].to_list()

src_host_indices = torch.tensor([host_id_map[ip] for ip in src_ids])
dst_host_indices = torch.tensor([host_id_map[ip] for ip in dst_ids])

flow_indices = torch.arange(num_flows)

src_edge_index = torch.stack([src_host_indices, flow_indices], dim=0)
dst_edge_index = torch.stack([dst_host_indices, flow_indices], dim=0)

# %%
data = HeteroData()
data['host'].x = host_features
data['flow'].x = flow_features

y = y['Label']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.to_list())
data['flow'].y = torch.tensor(y_encoded, dtype=torch.long)


# Edge indices
data['host', 'src_of', 'flow'].edge_index = src_edge_index
data['host', 'dst_of', 'flow'].edge_index = dst_edge_index

# Optional reverse edges
data['flow', 'rev_src_of', 'host'].edge_index = src_edge_index.flip(0)
data['flow', 'rev_dst_of', 'host'].edge_index = dst_edge_index.flip(0)

# %%


def split_masks(data, split=(0.7, 0.15, 0.15), seed=42):
    torch.manual_seed(seed)
    num_nodes = data['flow'].num_nodes
    perm = torch.randperm(num_nodes)

    train_end = int(split[0] * num_nodes)
    val_end = train_end + int(split[1] * num_nodes)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    for name, idx in [('train_mask', train_idx), ('val_mask', val_idx), ('test_mask', test_idx)]:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        data['flow'][name] = mask


# %%

# %%

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = HeteroConv({
            ('host', 'src_of', 'flow'): GATConv(-1, hidden_channels, add_self_loops=False),
            ('host', 'dst_of', 'flow'): GATConv(-1, hidden_channels, add_self_loops=False),
            ('flow', 'rev_src_of', 'host'): GATConv(-1, hidden_channels, add_self_loops=False),
            ('flow', 'rev_dst_of', 'host'): GATConv(-1, hidden_channels, add_self_loops=False),
        }, aggr='sum')
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return self.lin(x_dict['flow'])

# %%


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['flow'].train_mask], data['flow'].y[data['flow'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data['flow'].y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        return acc


# %%


def objective(trial):
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = HeteroGNN(data.metadata(), hidden_channels, len(torch.unique(data['flow'].y)), dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=f"runs/trial_{trial.number}")
    best_val = 0

    for epoch in range(1, 51):
        loss = train(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, data['flow'].val_mask)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        best_val = max(best_val, val_acc)

    writer.close()
    return best_val


# %%
data['host'].x = torch.nn.Embedding(num_hosts, 32).weight

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
split_masks(data)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best hyperparams:", study.best_params)
