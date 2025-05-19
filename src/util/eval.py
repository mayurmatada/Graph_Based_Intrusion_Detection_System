from src.models.GNN import process_and_load_data, split_masks, HeteroGNN, torch_config, evaluate_f1
import torch
import logging
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize

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
    """
    Loads a pre-trained HeteroGNN model from a checkpoint file and prepares it for evaluation.

    Args:
        device (torch.device): The device to which the model should be moved (e.g., 'cpu' or 'cuda').
        data (torch_geometric.data.HeteroData): The heterogeneous graph data object containing metadata.
        num_hosts (int): The number of host nodes in the graph.
        flow_features (int): The number of features for the 'flow' node type.
        file (str): The filename of the checkpoint to load.

    Returns:
        HeteroGNN: The loaded model set to evaluation mode and moved to the specified device.
    """
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
    """
    Loads a model and evaluates its F1 score on the test dataset.

    Args:
        device (torch.device or str): The device to load the model onto (e.g., 'cpu' or 'cuda').
        num_hosts (int): The number of hosts in the dataset.
        flow_features (int): The number of features per network flow.
        data (object): The dataset or data object required by the model and evaluation functions.
        file (str): Path to the model file to be loaded.

    Returns:
        tuple: A tuple containing:
            - F1_test (float): The F1 score of the model evaluated on the test set.
            - model (object): The loaded model instance.
    """
    model = load_model(device, data, num_hosts, flow_features, file)
    F1_test = evaluate_f1(model, data, 'test')
    return F1_test, model


def generate_confusion_matrix(model, data, device, class_names=None):
    """
    Generates and saves a normalized confusion matrix heatmap for model predictions on the test set.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data (torch_geometric.data.HeteroData): The input data containing node features, edge indices, and masks.
        device (torch.device): The device (CPU or CUDA) to run the model and data on.
        class_names (list of str, optional): List of class names for labeling the confusion matrix axes. 
            If None, unique labels from the data are used.

    Side Effects:
        - Saves the confusion matrix heatmap as 'Parameter_Databases/Graphs/confusion_matrix.png'.
        - Logs the save operation.

    Notes:
        - The confusion matrix is row-normalized to show percentages.
        - Only test set samples (where test_mask is True) are evaluated.
    """
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
    """
    Generates and saves a ROC curve plot for a multi-class classification model.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data (torch_geometric.data.HeteroData): The input data containing features, edge indices, and labels.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        class_names (list of str, optional): List of class names for labeling the ROC curves. If None, class indices are used.

    Saves:
        A ROC curve plot as 'Parameter_Databases/Graphs/roc_curve.png'.

    Logs:
        An info message indicating the ROC curve has been saved.
    """
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
    """
    Generates and saves a precision-recall curve for each class using the provided model and data.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data (torch_geometric.data.HeteroData): The input data containing features, labels, and masks.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        class_names (list of str, optional): List of class names for labeling the curves. If None, class indices are used.

    Saves:
        A PNG image of the precision-recall curves for all classes at 'Parameter_Databases/Graphs/precision_recall_curve.png'.

    Logs:
        An info message indicating the location of the saved precision-recall curve.
    """
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

    else:
        logger.error("No valid model found for visualization.")


if __name__ == "__main__":
    main()
