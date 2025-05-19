from src.models.GNN import process_and_load_data, split_masks, HeteroGNN, torch_config, evaluate_f1
import torch
import logging
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    plt.savefig("Graphs/confusion_matrix.png")
    logger.info("Saved confusion matrix as 'confusion_matrix.png'")


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
    generate_confusion_matrix(best_model, data, device, class_names=encoder.classes_)


if __name__ == "__main__":
    main()
