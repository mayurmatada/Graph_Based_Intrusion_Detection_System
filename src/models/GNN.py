from colorlog import ColoredFormatter
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
from torch.nn import Linear, Embedding
from torch_geometric.nn import SAGEConv, HeteroConv
import optuna
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import HeteroData
import gc
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import warnings


def logging_and_torch_config():
    """
    Initializes the environment for running GNN experiments.

    This function performs the following setup steps:
    - Suppresses experimental warnings from Optuna.
    - Ensures the existence of a 'logs' directory for log files.
    - Configures logging to output both to a timestamped log file and the console, with color formatting for console output.
    - Sets up the PyTorch device (CUDA if available, otherwise CPU).
    - Sets the matrix multiplication precision for PyTorch to 'high'.
    - Sets a manual random seed for PyTorch for reproducibility.

    Returns:
        torch.device: The device (CUDA or CPU) to be used for PyTorch operations.
    """
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# Ensure log directory exists
    os.makedirs("logs", exist_ok=True)

# Timestamped log file
    log_file = datetime.now().strftime("logs/run_%Y%m%d_%H%M%S.log")

# ----------------------------
# FORMATTERS
# ----------------------------
    plain_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    color_format = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

# ----------------------------
# HANDLERS
# ----------------------------
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(plain_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_format)

# ----------------------------
# LOGGER
# ----------------------------
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    device = torch_config()

    logging.basicConfig(
        level=logging.INFO,                      # Set logging level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Optional format
        handlers=[                               # Explicitly set handlers
            logging.StreamHandler()
        ]
    )

    return device


def torch_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)
    return device


# Load data


def process_and_load_data(device):
    """
    Loads processed network flow data, prepares features and labels, and constructs a heterogeneous graph for GNN models.

    Args:
        device (torch.device): The device (CPU or CUDA) to which tensors and data should be moved.

    Returns:
        num_hosts (int): Number of unique hosts (source and destination IPs) in the dataset.
        flow_features (torch.Tensor): Tensor of flow feature vectors (excluding IPs and ports), shape (num_flows, num_features).
        data (HeteroData): PyTorch Geometric HeteroData object representing the heterogeneous graph with hosts and flows.
        encoder (LabelEncoder): Fitted LabelEncoder instance for encoding flow labels.
        y_encoded (np.ndarray): Encoded labels for each flow as a NumPy array.

    Notes:
        - Expects processed feature and label parquet files at 'data/Processed_and_split/Processed_X.parquet' and 'data/Processed_and_split/Processed_y.parquet'.
        - Constructs a bipartite graph between hosts and flows, with edge types for source and destination relationships.
        - Host and flow nodes are connected via 'src_of', 'dst_of', and their reverse edge types.
    """
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
    return num_hosts, flow_features, data, encoder, y_encoded


# Masks


def split_masks(data, device, split=(0.7, 0.15, 0.15), seed=42):
    """
    Splits the nodes of the 'flow' graph in the input data into training, validation, and test sets,
    and creates corresponding boolean masks.

    Args:
        data (dict): A dictionary containing a 'flow' key, whose value is a graph object with a 'num_nodes' attribute.
        split (tuple, optional): A tuple of three floats indicating the proportion of nodes for training, validation, and test sets, respectively. Defaults to (0.7, 0.15, 0.15).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Modifies:
        data['flow']: Adds 'train_mask', 'val_mask', and 'test_mask' boolean attributes to the 'flow' graph,
        indicating the membership of each node in the respective set.
    """
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


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-class classification tasks.

    Focal Loss is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives.
    This implementation supports optional class weighting (alpha), focusing parameter (gamma), and configurable reduction.

    Args:
        alpha (Tensor or None): Optional tensor of shape [num_classes] specifying per-class weights. Default is None.
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted. Default is 2.0.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.

    Shape:
        - input: (N, C) where C = number of classes.
        - target: (N) where each value is 0 ≤ targets[i] ≤ C−1.

    Returns:
        Tensor: Loss value according to the specified reduction.

    References:
        - Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). 
          Focal Loss for Dense Object Detection. https://arxiv.org/abs/1708.02002
    """

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
    """
    A heterogeneous Graph Neural Network (GNN) model for graphs with 'host' and 'flow' node types.

    This model uses node embeddings for hosts, two layers of heterogeneous SAGEConv convolutions,
    batch normalization, ReLU activations, clamping, and dropout for regularization. It is designed
    to process graphs with multiple node and edge types, such as those found in network intrusion
    detection systems.

    Args:
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]):
            Metadata describing node types and edge types in the heterogeneous graph.
        hidden_channels (int):
            Number of hidden units in the convolutional layers.
        out_channels (int):
            Number of output units for the final linear layer.
        dropout (float):
            Dropout probability applied after activations.
        num_hosts (int):
            Number of unique host nodes (for embedding initialization).
        embedding_dim (int, optional):
            Dimensionality of the host node embeddings. Default is 16.

    Inputs:
        x_dict (Dict[str, Tensor]):
            Dictionary mapping node types to their feature tensors.
            - 'host': Will be replaced by learned embeddings.
            - 'flow': Input features for flow nodes.
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]):
            Dictionary mapping edge types to their edge index tensors.

    Returns:
        Dict[str, Tensor]:
            Dictionary containing the output logits for 'flow' nodes:
            - 'flow': Tensor of shape [num_flow_nodes, out_channels]
    """

    def __init__(self, metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim=16):
        super().__init__()
        self.host_embedding = Embedding(num_hosts, embedding_dim)
        torch.nn.init.xavier_uniform_(self.host_embedding.weight)

        # BatchNorm for full batch training (more stable here)
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
        }, aggr='mean')

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((hidden_channels, hidden_channels), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Host embedding: keep float32, let AMP cast if needed
        x_dict['host'] = self.host_embedding.weight.to(device=device)

        # Flow input: keep float32, normalization + clamp
        x_dict['flow'] = x_dict['flow'].to(device=device)
        x_dict['flow'] = (x_dict['flow'] - 0) / (1 + 1e-8)  # mean=0 std=1
        x_dict['flow'] = torch.clamp(x_dict['flow'], -10, 10)

        # First heterogeneous conv
        x_dict = self.conv1(x_dict, edge_index_dict)

        # BatchNorm1 + ReLU + clamp + Dropout (BatchNorm in float32)
        for k in x_dict:
            x = x_dict[k].to(torch.float32)  # ensure stable BatchNorm
            x = self.norm1[k](x)
            x = x.to(dtype)  # back to AMP dtype (float16) if used
            x = F.relu(x)
            x = torch.clamp(x, -1e2, 1e2)
            x_dict[k] = F.dropout(x, p=self.dropout, training=self.training)

        # Second heterogeneous conv
        x_dict = self.conv2(x_dict, edge_index_dict)

        # BatchNorm2 + ReLU + clamp + Dropout (BatchNorm in float32)
        for k in x_dict:
            x = x_dict[k].to(torch.float32)
            x = self.norm2[k](x)
            x = x.to(dtype)
            x = F.relu(x)
            x = torch.clamp(x, -1e2, 1e2)
            x_dict[k] = F.dropout(x, p=self.dropout, training=self.training)

        # Final linear layer on flow nodes
        out = self.lin(x_dict['flow'])
        return {'flow': out}


def train(model, data, optimizer, criterion, scaler):
    """
    Trains the given model for one iteration on the provided data batch.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        data (torch_geometric.data.HeteroData): Input batch containing node features, edge indices, and masks.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (callable): Loss function to compute the training loss.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.

    Returns:
        float: The computed loss value for the current training iteration.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}

    with autocast("cuda"):
        out = model(x_dict, data.edge_index_dict)

        loss = criterion(out['flow'][data['flow'].train_mask], data['flow'].y[data['flow'].train_mask])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def evaluate(model, data, split='val'):
    """
    Evaluates the performance of a GNN model on a specified data split.

    Args:
        model (torch.nn.Module): The GNN model to evaluate.
        data (torch_geometric.data.HeteroData): The input data containing node features, edge indices, and masks.
        split (str, optional): The data split to evaluate on ('val', 'test', etc.). Defaults to 'val'.

    Returns:
        float: The accuracy of the model on the specified split.
    """
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)

    mask = data['flow'][f"{split}_mask"]
    pred = out['flow'][mask].argmax(dim=1)
    acc = (pred == data['flow'].y[mask]).float().mean().item()
    return acc


def evaluate_f1(model, data, split='val'):
    """
    Evaluates the macro-averaged F1 score of a GNN model on a specified data split.

    Args:
        model (torch.nn.Module): The graph neural network model to evaluate.
        data (torch_geometric.data.HeteroData): The heterogeneous graph data containing node features, edge indices, and masks.
        split (str, optional): The data split to evaluate on ('train', 'val', or 'test'). Defaults to 'val'.

    Returns:
        float: The macro-averaged F1 score for the specified split.
    """
    device = torch_config()
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)
        mask = data['flow'][f"{split}_mask"]
        preds = out['flow'][mask].argmax(dim=1).cpu()
        labels = data['flow'].y[mask].cpu()
        return f1_score(labels, preds, average='macro')


def evaluate_accuracy(model, data, split='val'):
    """
    Evaluates the accuracy of a GNN model on a specified data split.

    Args:
        model (torch.nn.Module): The GNN model to evaluate.
        data (torch_geometric.data.HeteroData): The heterogeneous graph data containing node features, edge indices, and masks.
        split (str, optional): The data split to evaluate on ('train', 'val', or 'test'). Defaults to 'val'.

    Returns:
        float: The accuracy score of the model predictions on the specified split.
    """
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        out = model(x_dict, data.edge_index_dict)
        mask = data['flow'][f"{split}_mask"]
        preds = out['flow'][mask].argmax(dim=1).cpu()
        labels = data['flow'].y[mask].cpu()
        return accuracy_score(labels, preds)


def objective(trial):
    """
    Objective function for hyperparameter optimization of a Heterogeneous Graph Neural Network (GNN) using Optuna.

    This function samples hyperparameters, trains the GNN model, evaluates its performance, and implements early stopping based on F1 score stagnation. It also logs training metrics and hyperparameters to TensorBoard, saves the best model checkpoint, and manages resources.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest hyperparameters.

    Returns:
        float: The best validation F1 score achieved during training.

    Hyperparameters Tuned:
        - hidden_channels (int): Number of hidden channels in the GNN.
        - dropout (float): Dropout rate.
        - lr (float): Learning rate for the optimizer.
        - gamma (float): Focusing parameter for the Focal Loss.

    Side Effects:
        - Writes TensorBoard logs for training loss, validation F1, and accuracy.
        - Saves the best model checkpoint to disk.
        - Logs hyperparameters and final metrics to TensorBoard.
        - Clears CUDA cache and collects garbage after training.

    Early Stopping:
        - Stops training if the maximum F1 score improvement within a sliding window of epochs is less than a minimum threshold.
    """
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32])
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    gamma = trial.suggest_float('gamma', 0.5, 5.0)

    model = HeteroGNN(
        data.metadata(), hidden_channels,
        len(torch.unique(data['flow'].y)),
        dropout,
        num_hosts=num_hosts,
        flow_features=flow_features
    )
    model = model.to(device)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_encoded),
                                         y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)

    writer = SummaryWriter(log_dir=f"Parameter_Databases/Tensorboard/{trial.number}")

    best_val_f1 = 0
    best_val_acc = 0
    best_epoch = 0
    max_epochs = 400
    scaler = GradScaler("cuda")
    best_model_state = None
    f1_history = []

    patience_window = 30
    min_improvement = 0.001

    for epoch in range(1, max_epochs + 1):
        loss = train(model, data, optimizer, criterion, scaler)
        val_f1 = evaluate_f1(model, data, split='val')
        val_acc = evaluate_accuracy(model, data, split='val')

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        f1_history.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()

        # Maintain sliding window of last N F1s
        if len(f1_history) > patience_window or best_val_f1 > 0.95:
            f1_history.pop(0)
            if max(f1_history) - min(f1_history) < min_improvement:
                logging.info(f"Early stopping at epoch {epoch} (F1 stagnated < {min_improvement} over {patience_window} epochs)")
                break
            elif best_val_f1 > 0.95:
                logging.info(f"Early stopping at epoch {epoch} (F1 cut at 0.95)")
                break

        logging.info(f"Trial {trial.number} Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")
    if best_model_state is not None:
        checkpoint_dir = "Parameter_Databases/Checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'lr': lr,
            'gamma': gamma,
            'best_epoch': best_epoch
        }, f"{checkpoint_dir}/trial_{trial.number}.pt")

    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'lr': lr,
            'gamma': gamma
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


def optuna_optimize(data, split_masks, objective, device):
    """
    Optimizes hyperparameters for a GNN model using Optuna.

    This function splits the input data using the provided split_masks function, sets up an Optuna study with a SQLite backend, and runs the optimization process for a single trial using the given objective function. The best hyperparameters found are logged.

    Args:
        data: The dataset to be used for training and validation.
        split_masks (Callable): A function that splits the data into training, validation, and test sets.
        objective (Callable): The objective function to be optimized by Optuna.

    Returns:
        optuna.study.Study: The Optuna study object containing the optimization results.
    """
    split_masks(data, device)

    storage = "sqlite:///Parameter_Databases/Optuna/optuna_study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name="hetero_gnn_intrusion",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1)

    logging.info(f"Best hyperparameters: {study.best_params}")
    return study


if __name__ == "__main__":
    device = logging_and_torch_config()
    num_hosts, flow_features, data, encoder, y_encoded = process_and_load_data(device)
    split_masks(data, device)
    study = optuna_optimize(data, split_masks, objective, device)
