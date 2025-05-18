import time
from sklearn.metrics import accuracy_score
from torch.nn import Linear, Embedding
from torch_geometric.nn import SAGEConv, HeteroConv
import optuna
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import HeteroData
import gc
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)


# Load data
X = pl.read_parquet("data/Processed_and_split/Processed_X.parquet")
y = pl.read_parquet("data/Processed_and_split/Processed_y.parquet")

src_ips = X['Source IP'].unique()
dest_ips = X['Destination IP'].unique()
unique_hosts = list(set(src_ips) | set(dest_ips))
host_id_map = {v: k for k, v in enumerate(unique_hosts)}
num_hosts = len(unique_hosts)

embedding_dim = 32
# host_embedding = torch.nn.Embedding(num_hosts, embedding_dim).to(device)

flow_features_df = X.drop(['Source IP', 'Destination IP', 'Source Port', 'Destination Port'])
flow_features = torch.tensor(flow_features_df.to_numpy(), dtype=torch.float32).to(device)
num_flows = flow_features.shape[0]

src_ids = X["Source IP"].to_list()
dst_ids = X["Destination IP"].to_list()

src_host_indices = torch.tensor([host_id_map[ip] for ip in src_ids], device=device)
dst_host_indices = torch.tensor([host_id_map[ip] for ip in dst_ids], device=device)
flow_indices = torch.arange(num_flows, device=device)

src_edge_index = torch.stack([src_host_indices, flow_indices], dim=0)
dst_edge_index = torch.stack([dst_host_indices, flow_indices], dim=0)

data = HeteroData()

data['flow'].x = flow_features


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y['Label'].to_list())
data['flow'].y = torch.tensor(y_encoded, dtype=torch.long, device=device)

data['host', 'src_of', 'flow'].edge_index = src_edge_index
data['host', 'dst_of', 'flow'].edge_index = dst_edge_index
data['flow', 'rev_src_of', 'host'].edge_index = src_edge_index.flip(0)
data['flow', 'rev_dst_of', 'host'].edge_index = dst_edge_index.flip(0)

data = data.to(device)

# Masks


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
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[idx] = True
        data['flow'][name] = mask


split_masks(data)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # tensor of shape [num_classes] or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, dropout, num_hosts, embedding_dim=32):
        super().__init__()
        self.host_embedding = Embedding(num_hosts, embedding_dim)
        torch.nn.init.xavier_uniform_(self.host_embedding.weight)

        self.norm1 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in ['flow', 'host']
        })

        self.norm2 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in ['flow', 'host']
        })

        self.conv1 = HeteroConv({
            ('host', 'src_of', 'flow'): SAGEConv((embedding_dim, flow_features.shape[1]), hidden_channels),
            ('host', 'dst_of', 'flow'): SAGEConv((embedding_dim, flow_features.shape[1]), hidden_channels),
            ('flow', 'rev_src_of', 'host'): SAGEConv((flow_features.shape[1], embedding_dim), hidden_channels),
            ('flow', 'rev_dst_of', 'host'): SAGEConv((flow_features.shape[1], embedding_dim), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((hidden_channels, hidden_channels), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        x_dict['host'] = self.host_embedding.weight
        x_dict['flow'] = x_dict['flow'].to(device=device, dtype=dtype)

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.norm1[k](v) for k, v in x_dict.items()}
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.norm2[k](v) for k, v in x_dict.items()}
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        out = self.lin(x_dict['flow'])
        return {'flow': out}


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}

    out = model(x_dict, data.edge_index_dict)
    loss = criterion(out['flow'][data['flow'].train_mask], data['flow'].y[data['flow'].train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, data, split='val'):
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)

    mask = data['flow'][f"{split}_mask"]
    pred = out['flow'][mask].argmax(dim=1)
    acc = (pred == data['flow'].y[mask]).float().mean().item()
    return acc


def evaluate_f1(model, data, split='val'):
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)
        mask = data['flow'][f"{split}_mask"]
        preds = out['flow'][mask].argmax(dim=1).cpu()
        labels = data['flow'].y[mask].cpu()
        return f1_score(labels, preds, average='macro')


def evaluate_accuracy(model, data, split='val'):
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)
        mask = data['flow'][f"{split}_mask"]
        preds = out['flow'][mask].argmax(dim=1).cpu()
        labels = data['flow'].y[mask].cpu()
        return accuracy_score(labels, preds)


def objective(trial):
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = HeteroGNN(
        data.metadata(), hidden_channels,
        len(torch.unique(data['flow'].y)),
        dropout,
        num_hosts=num_hosts
    ).to(device)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_encoded),
                                         y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    writer = SummaryWriter(log_dir=f"Parameter_Databases/Tensorboard/{trial.number}")

    best_val_f1 = 0
    best_val_acc = 0
    best_epoch = 0
    patience = 100
    max_epochs = 400

    for epoch in range(1, max_epochs + 1):
        loss = train(model, data, optimizer, criterion)
        val_f1 = evaluate_f1(model, data, split='val')
        val_acc = evaluate_accuracy(model, data, split='val')

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            logging.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        logging.info(f"Trial {trial.number} Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")

    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'lr': lr
        },
        {
            'hparam/val_f1': best_val_f1,
            'hparam/val_acc': best_val_acc
        }
    )

    writer.flush()
    writer.close()

    del model, optimizer, criterion, writer
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_f1


split_masks(data)

storage = "sqlite:///Parameter_Databases/Optuna/optuna_study.db"

study = optuna.create_study(
    direction="maximize",
    study_name="hetero_gnn_intrusion",
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=20)

print("Best hyperparams:", study.best_params)


def train_and_return_f1(best_params, model_class, data, device='cuda', num_epochs=200, show_report=True):
    class_names = encoder.classes_

    hidden_channels = best_params['hidden_channels']
    dropout = best_params['dropout']
    lr = best_params['lr']

    num_classes = len(torch.unique(data['flow'].y))
    num_hosts = data['host'].x.size(0) if 'host' in data.node_types else 0

    model = model_class(
        data.metadata(), hidden_channels, num_classes, dropout,
        num_hosts=num_hosts
    ).to(device)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_encoded),
                                         y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    writer = SummaryWriter(log_dir=f"Parameter_Databases/Tensorboard/final_model")

    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, split='val')
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

    writer.close()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['flow'].test_mask
        logits = out['flow'][mask]
        preds = logits.argmax(dim=1).cpu()
        labels = data['flow'].y[mask].cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())

    f1 = f1_score(y_true, y_pred, average='macro')

    if show_report:
        print("=== Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

        logging.info("=== Final Evaluation on Test Set ===")
        logging.info(f"Macro F1 Score: {f1:.4f}")

    return model, f1


model, test_f1 = train_and_return_f1(
    best_params=study.best_params,
    model_class=HeteroGNN,
    data=data,
    device='cuda'

)
logging.info(f"Best hyperparameters: {study.best_params}")
