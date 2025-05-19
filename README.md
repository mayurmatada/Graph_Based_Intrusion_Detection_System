# GNN-based Intrusion Detection on CICIDS 2017
This project implements a Graph Neural Network (GNN) for detecting
network intrusions using the CICIDS 2017 dataset.
The model leverages a heterogeneous graph representation to capture
both host and flow-level behavior, enabling robust classification
of malicious traffic patterns.
The graph is constructed with two types of nodes:
- **Flow nodes**: Represent individual network connections.
- **Host nodes**: Represent source and destination IPs. Edges connect flows to their corresponding hosts, allowing the model
to learn relationships between traffic and devices. The architecture
is designed using PyTorch Geometric , hyperparameter tuning with Optuna,
logging with TensorBoard, and checkpointing for reproducibility.

## How to Run
### 1. Clone the repository

```bash
git clone https://github.com/mayurmatada/Graph_Based_Intrusion_Detection_System
cd Graph_Based_Intrusion_Detection_System
```
### 2. Set up the Conda environment

```bash
conda env create -f gnnnet.yml
conda activate gnnnet
```

### 3. Download the CICIDS 2017 dataset

- Download the CICIDS 2017 CSV files from the official website:
  https://www.unb.ca/cic/datasets/ids-2017.html
- Place the raw CSV files in the following directory: data/Raw/

Ensure the files include:
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv `
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`

### 4. Run preprocessing

```bash
python src/util/Process.py
```
### 5. Train the GNN
```bash
bash Run_Training.sh
```
This runs the training loop, performs Optuna-based hyperparameter search,
and logs training metrics to:
- `logs/`
- `Parameter_Databases/Tensorboard/`
- `Parameter_Databases/Optuna/optuna_study.db`
- `Parameter_Databases/Checkpoints/`

### 6. Evaluate the GNN

```bash
bash Run_Evaluation.sh
```
This script loads the best checkpoint and evaluates it on the test set,
generating:
- ROC curves
- Precision-recall curves
- Confusion matrices

The plots are saved in `Parameter_Databases/Graphs/`.

## Notes
- Training and evaluation use PyTorch Geometric (`torch_geometric`)
and are designed for GPU execution.

- All model components, data processing utilities, and evaluation
scripts are modular and located under the `src/` directory.

- Logging is handled via Python’s built-in logging module,
and results are saved for reproducibility.

---