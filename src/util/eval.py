from matplotlib.lines import Line2D
import random
from src.models.GNN import process_and_load_data, split_masks, HeteroGNN, torch_config, evaluate_f1
import torch
import logging
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize
import networkx as nx

from torch_geometric.utils import to_networkx

# Colored logging formatter


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


def load_model(device, data, num_hosts, flow_features, file):
    logger.info(f"Loading model checkpoint: {file}")
    checkpoint = torch.load(f"Parameter_Databases/Checkpoints/{file}")
    model = HeteroGNN(
        data.metadata(), checkpoint['hidden_channels'],
        len(torch.unique(data['flow'].y)),
        checkpoint['dropout'],
        num_hosts=num_hosts,
        flow_features=flow_features
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


def get_F1s(device, num_hosts, flow_features, data, file):
    model = load_model(device, data, num_hosts, flow_features, file)
    F1_test = evaluate_f1(model, data, 'test')
    return F1_test, model


def generate_confusion_matrix(model, data, device, class_names=None):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

        preds = out['flow'].argmax(dim=1)
        y = data['flow'].y
        mask = data['flow'].test_mask

        # Ensure same device before indexing
        y = y[mask]
        preds = preds[mask]

        # Move to CPU
        y = y.cpu()
        preds = preds.cpu()

    if class_names is not None:
        labels = list(range(len(class_names)))

    else:
        labels = sorted(torch.unique(y).tolist())

    # Confusion matrix as raw counts
    cm = confusion_matrix(y, preds, labels=labels)

    # Convert to percentage (row-wise normalization)
    cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names)  # type: ignore
    plt.title("Confusion Matrix (Test Set) [%]")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Parameter_Databases/Graphs/confusion_matrix.png")
    logger.info("Saved confusion matrix as 'confusion_matrix.png'")


def generate_roc_curve(model, data, device, class_names=None):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = torch.softmax(out['flow'], dim=1)
        y_true = data['flow'].y
        mask = data['flow'].test_mask
        y_true = y_true[mask].cpu().numpy()
        probs = probs[mask].cpu().numpy()

    if class_names is None:
        class_names = [str(i) for i in range(probs.shape[1])]

    y_true_bin = label_binarize(y_true, classes=range(probs.shape[1]))

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # type: ignore
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title("ROC Curve (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Parameter_Databases/Graphs/roc_curve.png")
    plt.close()
    logger.info("Saved ROC curve as 'roc_curve.png'")


def generate_precision_recall_curve(model, data, device, class_names=None):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = torch.softmax(out['flow'], dim=1)
        y_true = data['flow'].y
        mask = data['flow'].test_mask
        y_true = y_true[mask].cpu().numpy()
        probs = probs[mask].cpu().numpy()

    if class_names is None:
        class_names = [str(i) for i in range(probs.shape[1])]

    y_true_bin = label_binarize(y_true, classes=range(probs.shape[1]))

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])  # type: ignore
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AUC = {pr_auc:.2f})")

    plt.title("Precision-Recall Curve (Test Set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Parameter_Databases/Graphs/precision_recall_curve.png")
    plt.close()
    logger.info("Saved Precision-Recall curve as 'precision_recall_curve.png'")


def visualize_hetero_graph(data, max_nodes=100, seed=42, show_labels=False):
    random.seed(seed)
    G = nx.Graph()

    # Assign unique colors to node types
    node_types = list(data.x_dict.keys())
    cmap_nodes = plt.cm.get_cmap("tab20", len(node_types))
    node_color_map = {ntype: cmap_nodes(i) for i, ntype in enumerate(node_types)}

    node_ids = {}
    for ntype in node_types:
        all_idxs = list(range(data[ntype].num_nodes))
        sampled_idxs = random.sample(all_idxs, min(max_nodes, len(all_idxs)))
        for idx in sampled_idxs:
            node_name = f"{ntype}_{idx}"
            G.add_node(node_name, ntype=ntype)
            node_ids[(ntype, idx)] = node_name

    # Edge type coloring
    edge_types = list(data.edge_index_dict.keys())
    cmap_edges = plt.cm.get_cmap("Dark2", len(edge_types))
    edge_color_map = {etype: cmap_edges(i) for i, etype in enumerate(edge_types)}
    edge_colors = []

    for etype in edge_types:
        src_type, rel_type, dst_type = etype
        edge_index = data[etype].edge_index
        for src, dst in edge_index.t().tolist():
            src_node = node_ids.get((src_type, src))
            dst_node = node_ids.get((dst_type, dst))
            if src_node and dst_node:
                G.add_edge(src_node, dst_node, rel_type=rel_type)
                edge_colors.append(edge_color_map[etype])

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Draw nodes
    for ntype in node_types:
        nodelist = [n for n, attr in G.nodes(data=True) if attr['ntype'] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                               node_color=[node_color_map[ntype]] * len(nodelist),  # type: ignore
                               label=ntype, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0, alpha=0.6)  # type: ignore

    # Draw labels (optional)
    if show_labels or len(G.nodes) <= 100:
        nx.draw_networkx_labels(G, pos, font_size=6)

    # Custom legends
    node_legend = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, label=ntype, markersize=6)
                   for ntype, color in node_color_map.items()]
    edge_legend = [Line2D([0], [0], color=color, label=f"{etype[1]}")
                   for etype, color in edge_color_map.items()]

    plt.legend(handles=node_legend + edge_legend, loc='best', fontsize='small')
    plt.title("Sampled Heterogeneous Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("Parameter_Databases/Graphs/Graph.png")


def main():
    model_checkpoints = os.listdir("Parameter_Databases/Checkpoints")
    logger.info("Initializing configuration...")
    device = torch_config()
    logger.info(f"Using device: {device}")

    logger.info("Processing and loading data...")
    num_hosts, flow_features, data, encoder, y_encoded = process_and_load_data(device)

    logger.info("Splitting data masks...")
    split_masks(data, device)

    best_f1 = -1
    best_model = None
    best_file = None

    for file in model_checkpoints:
        logger.info(f"Evaluating model: {file}")
        f1_score, model = get_F1s(device, num_hosts, flow_features, data, file)
        logger.info(f"F1 Score (test): {f1_score:.4f}")
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model = model
            best_file = file

    logger.info(f"Best model: {best_file} with F1: {best_f1:.4f}")

    if best_model is not None:
        generate_confusion_matrix(best_model, data, device, class_names=encoder.classes_)
        generate_roc_curve(best_model, data, device, class_names=encoder.classes_)
        generate_precision_recall_curve(best_model, data, device, class_names=encoder.classes_)
        # visualize_hetero_graph(data)

    else:
        logger.error("No valid model found for visualization.")


if __name__ == "__main__":
    main()
