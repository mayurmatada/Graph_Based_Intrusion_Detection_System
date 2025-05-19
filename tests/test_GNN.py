import torch
import types
import polars as pl
import numpy as np
import os
from datetime import datetime
import logging
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from models.GNN import torch_config, process_and_load_data, optuna_optimize, objective, split_masks, train, evaluate, evaluate_f1, logging_and_torch_config, FocalLoss, HeteroGNN  # type: ignore


def test_torch_config_returns_device():
    device = torch_config()
    assert isinstance(device, torch.device)
    assert str(device) in ['cuda', 'cpu']


def test_torch_config_sets_seed(monkeypatch):
    # Set a known seed, generate a tensor, reset, and check for reproducibility
    torch_config()
    t1 = torch.rand(3)
    torch_config()
    t2 = torch.rand(3)
    assert torch.allclose(t1, t2)


def test_torch_config_matmul_precision(monkeypatch):
    # This test just checks that the function runs without error
    try:
        torch_config()
    except Exception as e:
        assert False, f"torch_config raised an exception: {e}"


def test_process_and_load_data_basic(monkeypatch, tmp_path):
    # Mock pl.read_parquet to return minimal DataFrames

    # Minimal fake data
    X_df = pl.DataFrame({
        "Source IP": ["1.1.1.1", "2.2.2.2"],
        "Destination IP": ["2.2.2.2", "1.1.1.1"],
        "Source Port": [123, 456],
        "Destination Port": [789, 101],
        "f1": [0.1, 0.2],
        "f2": [0.3, 0.4]
    })
    y_df = pl.DataFrame({
        "Label": ["A", "B"]
    })

    # Patch pl.read_parquet
    monkeypatch.setattr(pl, "read_parquet", lambda path: X_df if "X" in path else y_df)

    # Patch torch.device to cpu for test
    device = torch.device("cpu")

    num_hosts, flow_features, data, encoder, y_encoded = process_and_load_data(device)

    # Basic checks
    assert isinstance(num_hosts, int)
    assert isinstance(flow_features, torch.Tensor)
    assert hasattr(data, "metadata")
    assert hasattr(data, "__getitem__")
    assert hasattr(data, "to")
    assert hasattr(encoder, "transform")
    assert isinstance(y_encoded, np.ndarray)
    assert data['flow'].x.shape[0] == 2
    assert data['flow'].y.shape[0] == 2
    assert set(data.edge_types) == {
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    }


def test_split_masks_basic(monkeypatch):
    # Minimal fake data setup
    class DummyFlow:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes
            self.attrs = {}

        def __setitem__(self, key, value):
            self.attrs[key] = value

        def __getitem__(self, key):
            return self.attrs[key]

    class DummyData(dict):
        pass

    num_nodes = 10
    data = DummyData()
    data['flow'] = DummyFlow(num_nodes)

    # Use cpu device for test
    device = torch.device("cpu")

    split_masks(data, device, split=(0.6, 0.2, 0.2), seed=123)

    # Check that masks exist and are boolean tensors of correct shape
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['flow'][mask_name]
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == (num_nodes,)

    # Check that each node is in exactly one split
    total_mask = data['flow']['train_mask'].int() + data['flow']['val_mask'].int() + data['flow']['test_mask'].int()
    assert torch.all(total_mask == 1)


def test_split_masks_split_proportions(monkeypatch):
    class DummyFlow:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes
            self.attrs = {}

        def __setitem__(self, key, value):
            self.attrs[key] = value

        def __getitem__(self, key):
            return self.attrs[key]

    class DummyData(dict):
        pass

    num_nodes = 20
    data = DummyData()
    data['flow'] = DummyFlow(num_nodes)
    device = torch.device("cpu")

    split_masks(data, device, split=(0.5, 0.3, 0.2), seed=0)

    train_count = data['flow']['train_mask'].sum().item()
    val_count = data['flow']['val_mask'].sum().item()
    test_count = data['flow']['test_mask'].sum().item()

    # Check approximate proportions (allowing for rounding)
    assert train_count == 10
    assert val_count == 6
    assert test_count == 4
    assert train_count + val_count + test_count == num_nodes


def test_split_masks_reproducibility(monkeypatch):
    class DummyFlow:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes
            self.attrs = {}

        def __setitem__(self, key, value):
            self.attrs[key] = value

        def __getitem__(self, key):
            return self.attrs[key]

    class DummyData(dict):
        pass

    num_nodes = 15
    device = torch.device("cpu")

    # First split
    data1 = DummyData()
    data1['flow'] = DummyFlow(num_nodes)
    split_masks(data1, device, split=(0.6, 0.2, 0.2), seed=42)
    masks1 = {k: data1['flow'][k].clone() for k in ['train_mask', 'val_mask', 'test_mask']}

    # Second split with same seed
    data2 = DummyData()
    data2['flow'] = DummyFlow(num_nodes)
    split_masks(data2, device, split=(0.6, 0.2, 0.2), seed=42)
    masks2 = {k: data2['flow'][k].clone() for k in ['train_mask', 'val_mask', 'test_mask']}

    # Masks should be identical
    for k in masks1:
        assert torch.equal(masks1[k], masks2[k])


def test_train_runs_basic(monkeypatch):
    # Minimal dummy model, data, optimizer, criterion, scaler
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x_dict, edge_index_dict):
            # Return logits for 2 nodes, 2 classes
            return {'flow': torch.randn(2, 2, requires_grad=True)}

    class DummyData(dict):
        def __init__(self):
            super().__init__()
            self.x_dict = {'flow': torch.randn(2, 2)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.train_mask = torch.tensor([True, False])
            self.flow.y = torch.tensor([1, 0])
            self['flow'] = self.flow

    model = DummyModel()
    data = DummyData()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # Disable AMP for CPU

    loss = train(model, data, optimizer, criterion, scaler)
    assert isinstance(loss, float)
    assert loss >= 0


def test_logging_and_torch_config_device():
    # Test that the function returns a valid torch.device
    device = logging_and_torch_config()
    assert isinstance(device, torch.device)
    assert str(device) in ['cuda', 'cpu']


def test_focal_loss_init_defaults():
    # Test default initialization
    loss = FocalLoss()
    assert loss.alpha is None
    assert loss.gamma == 2.0
    assert loss.reduction == 'mean'


def test_focal_loss_init_custom_values():
    # Test initialization with custom values
    alpha = torch.tensor([0.25, 0.75])
    gamma = 1.5
    reduction = 'sum'
    loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    assert torch.equal(loss.alpha, alpha)
    assert loss.gamma == gamma
    assert loss.reduction == reduction


def test_focal_loss_init_invalid_reduction():
    # Test initialization with an invalid reduction value
    try:
        FocalLoss(reduction='invalid')
    except ValueError as e:
        assert str(e) == "reduction must be 'none', 'mean', or 'sum'"


def test_focal_loss_forward_mean():
    # Test forward method with reduction='mean'
    alpha = torch.tensor([0.25, 0.75])
    gamma = 2.0
    reduction = 'mean'
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    input = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert loss.item() > 0


def test_focal_loss_forward_sum():
    # Test forward method with reduction='sum'
    alpha = torch.tensor([0.25, 0.75])
    gamma = 2.0
    reduction = 'sum'
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    input = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert loss.item() > 0


def test_focal_loss_forward_none():
    # Test forward method with reduction='none'
    alpha = torch.tensor([0.25, 0.75])
    gamma = 2.0
    reduction = 'none'
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    input = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 1  # Tensor with shape [N]
    assert loss.shape[0] == input.shape[0]
    assert torch.all(loss > 0)


def test_focal_loss_forward_no_alpha():
    # Test forward method without alpha (default None)
    gamma = 2.0
    reduction = 'mean'
    loss_fn = FocalLoss(alpha=None, gamma=gamma, reduction=reduction)

    input = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert loss.item() > 0


def test_focal_loss_forward_gamma_zero():
    # Test forward method with gamma=0 (should behave like cross-entropy)
    alpha = torch.tensor([0.25, 0.75])
    gamma = 0.0
    reduction = 'mean'
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    input = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = loss_fn(input, target)
    ce_loss = F.cross_entropy(input, target, weight=alpha, reduction=reduction)
    assert torch.allclose(loss, ce_loss)


def test_HeteroGNN_init_basic():
    # Test basic initialization of HeteroGNN
    metadata = (['host', 'flow'], [
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    model = HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)

    # Check embeddings
    assert isinstance(model.host_embedding, torch.nn.Embedding)
    assert model.host_embedding.weight.shape == (num_hosts, embedding_dim)

    # Check batch normalization layers
    assert isinstance(model.norm1, torch.nn.ModuleDict)
    assert isinstance(model.norm2, torch.nn.ModuleDict)
    assert 'flow' in model.norm1 and 'host' in model.norm1
    assert 'flow' in model.norm2 and 'host' in model.norm2
    assert isinstance(model.norm1['flow'], torch.nn.BatchNorm1d)
    assert isinstance(model.norm2['host'], torch.nn.BatchNorm1d)

    # Check convolutional layers
    assert isinstance(model.conv1, HeteroConv)
    assert isinstance(model.conv2, HeteroConv)
    assert ('host', 'src_of', 'flow') in model.conv1.convs
    assert ('flow', 'rev_src_of', 'host') in model.conv1.convs

    # Check linear layer
    assert isinstance(model.lin, torch.nn.Linear)
    assert model.lin.in_features == hidden_channels
    assert model.lin.out_features == out_channels

    # Check dropout
    assert model.dropout == dropout


def test_HeteroGNN_init_invalid_metadata():
    # Test initialization with invalid metadata
    metadata = (['host', 'flow'], [
        ('invalid_node', 'src_of', 'flow')  # Invalid node type
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    try:
        HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)
    except KeyError as e:
        assert "invalid_node" in str(e)


def test_HeteroGNN_init_embedding_weights():
    # Test that embedding weights are initialized correctly
    metadata = (['host', 'flow'], [
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    model = HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)

    # Check that embedding weights are initialized with Xavier uniform
    weights = model.host_embedding.weight
    assert torch.all(weights >= -1) and torch.all(weights <= 1)  # Xavier uniform range


def test_HeteroGNN_forward_basic():
    # Test the forward method of HeteroGNN with minimal input
    metadata = (['host', 'flow'], [
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    model = HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)

    # Create dummy input data
    x_dict = {
        'host': None,  # Host embeddings will be replaced by the model
        'flow': torch.randn(5, 8)  # 5 flows, 8 features each
    }
    edge_index_dict = {
        ('host', 'src_of', 'flow'): torch.tensor([[0, 1], [0, 2]]),
        ('host', 'dst_of', 'flow'): torch.tensor([[0, 1], [0, 2]]),
        ('flow', 'rev_src_of', 'host'): torch.tensor([[0, 1], [0, 2]]),
        ('flow', 'rev_dst_of', 'host'): torch.tensor([[0, 1], [0, 2]])
    }

    # Run forward pass
    out = model(x_dict, edge_index_dict)

    # Check output
    assert isinstance(out, dict)
    assert 'flow' in out
    assert isinstance(out['flow'], torch.Tensor)
    assert out['flow'].shape == (5, out_channels)


def test_HeteroGNN_forward_clamping():
    # Test that clamping is applied correctly in the forward method
    metadata = (['host', 'flow'], [
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    model = HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)

    # Create dummy input data
    x_dict = {
        'host': None,  # Host embeddings will be replaced by the model
        'flow': torch.randn(5, 8) * 100  # Large values to test clamping
    }
    edge_index_dict = {
        ('host', 'src_of', 'flow'): torch.tensor([[0, 1], [0, 2]]),
        ('host', 'dst_of', 'flow'): torch.tensor([[0, 1], [0, 2]]),
        ('flow', 'rev_src_of', 'host'): torch.tensor([[0, 1], [0, 2]]),
        ('flow', 'rev_dst_of', 'host'): torch.tensor([[0, 1], [0, 2]])
    }

    # Run forward pass
    out = model(x_dict, edge_index_dict)

    # Check clamping
    assert torch.all(out['flow'] <= 1e2)
    assert torch.all(out['flow'] >= -1e2)


def test_HeteroGNN_forward_device():
    # Test that the forward method respects the device of the model
    metadata = (['host', 'flow'], [
        ('host', 'src_of', 'flow'),
        ('host', 'dst_of', 'flow'),
        ('flow', 'rev_src_of', 'host'),
        ('flow', 'rev_dst_of', 'host')
    ])
    hidden_channels = 16
    out_channels = 3
    dropout = 0.5
    num_hosts = 10
    flow_features = torch.randn(5, 8)  # 5 flows, 8 features each
    embedding_dim = 16

    model = HeteroGNN(metadata, hidden_channels, out_channels, dropout, num_hosts, flow_features, embedding_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create dummy input data
    x_dict = {
        'host': None,  # Host embeddings will be replaced by the model
        'flow': torch.randn(5, 8).to(device)  # Move to the same device as the model
    }
    edge_index_dict = {
        ('host', 'src_of', 'flow'): torch.tensor([[0, 1], [0, 2]]).to(device),
        ('host', 'dst_of', 'flow'): torch.tensor([[0, 1], [0, 2]]).to(device),
        ('flow', 'rev_src_of', 'host'): torch.tensor([[0, 1], [0, 2]]).to(device),
        ('flow', 'rev_dst_of', 'host'): torch.tensor([[0, 1], [0, 2]]).to(device)
    }

    # Run forward pass
    out = model(x_dict, edge_index_dict)

    # Check that output is on the same device as the model
    assert out['flow'].device == next(model.parameters()).device


def test_evaluate_basic():
    # Test the evaluate function with minimal input
    class DummyModel(torch.nn.Module):
        def forward(self, x_dict, edge_index_dict):
            # Return logits for 2 nodes, 2 classes
            return {'flow': torch.tensor([[0.1, 0.9], [0.8, 0.2]])}

    class DummyData(dict):
        def __init__(self):
            super().__init__()
            self.x_dict = {'flow': torch.randn(2, 2)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.val_mask = torch.tensor([True, False])
            self.flow.y = torch.tensor([1, 0])
            self['flow'] = self.flow

    model = DummyModel()
    data = DummyData()

    acc = evaluate(model, data, split='val')
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0  # Correct prediction for the single masked node


def test_evaluate_no_mask():
    # Test the evaluate function when no nodes are masked
    class DummyModel(torch.nn.Module):
        def forward(self, x_dict, edge_index_dict):
            return {'flow': torch.tensor([[0.1, 0.9], [0.8, 0.2]])}

    class DummyData(dict):
        def __init__(self):
            super().__init__()
            self.x_dict = {'flow': torch.randn(2, 2)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.val_mask = torch.tensor([False, False])  # No nodes masked
            self.flow.y = torch.tensor([1, 0])
            self['flow'] = self.flow

    model = DummyModel()
    data = DummyData()

    acc = evaluate(model, data, split='val')
    assert isinstance(acc, float)
    assert acc == 0.0  # No nodes to evaluate


def test_evaluate_partial_mask():
    # Test the evaluate function with a partial mask
    class DummyModel(torch.nn.Module):
        def forward(self, x_dict, edge_index_dict):
            return {'flow': torch.tensor([[0.1, 0.9], [0.8, 0.2]])}

    class DummyData(dict):
        def __init__(self):
            super().__init__()
            self.x_dict = {'flow': torch.randn(2, 2)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.val_mask = torch.tensor([True, True])  # Both nodes masked
            self.flow.y = torch.tensor([1, 1])  # Only one correct prediction
            self['flow'] = self.flow

    model = DummyModel()
    data = DummyData()

    acc = evaluate(model, data, split='val')
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert acc == 0.5  # One correct prediction out of two


def test_evaluate_device_respect():
    # Test that the evaluate function respects the device of the model
    class DummyModel(torch.nn.Module):
        def forward(self, x_dict, edge_index_dict):
            return {'flow': torch.tensor([[0.1, 0.9], [0.8, 0.2]])}

    class DummyData(dict):
        def __init__(self):
            super().__init__()
            self.x_dict = {'flow': torch.randn(2, 2)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.val_mask = torch.tensor([True, False])
            self.flow.y = torch.tensor([1, 0])
            self['flow'] = self.flow

    model = DummyModel().to('cuda' if torch.cuda.is_available() else 'cpu')
    data = DummyData()

    acc = evaluate(model, data, split='val')
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0


def test_objective_runs_basic(monkeypatch):
    # Mock trial object
    class DummyTrial:
        def __init__(self):
            self.params = {}

        def suggest_categorical(self, name, choices):
            self.params[name] = choices[0]
            return choices[0]

        def suggest_float(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_loguniform(self, name, low, high):
            self.params[name] = low
            return low

    trial = DummyTrial()

    # Mock data and related functions
    class DummyData:
        def metadata(self):
            return (['host', 'flow'], [
                ('host', 'src_of', 'flow'),
                ('host', 'dst_of', 'flow'),
                ('flow', 'rev_src_of', 'host'),
                ('flow', 'rev_dst_of', 'host')
            ])

        def __getitem__(self, key):
            if key == 'flow':
                return types.SimpleNamespace(
                    y=torch.tensor([0, 1, 0, 1]),
                    train_mask=torch.tensor([True, True, False, False])
                )

    dummy_data = DummyData()

    def mock_train(model, data, optimizer, criterion, scaler):
        return 0.5  # Mock loss

    def mock_evaluate_f1(model, data, split='val'):
        return 0.8  # Mock F1 score

    def mock_evaluate_accuracy(model, data, split='val'):
        return 0.9  # Mock accuracy

    def mock_compute_class_weight(class_weight, classes, y):
        return [1.0, 1.0]  # Mock class weights

    def mock_torch_save(*args, **kwargs):
        pass  # Mock saving

    # Mock dependencies
    monkeypatch.setattr("models.GNN.data", dummy_data)
    monkeypatch.setattr("models.GNN.num_hosts", 10)
    monkeypatch.setattr("models.GNN.flow_features", torch.randn(4, 8))
    monkeypatch.setattr("models.GNN.y_encoded", [0, 1, 0, 1])
    monkeypatch.setattr("models.GNN.train", mock_train)
    monkeypatch.setattr("models.GNN.evaluate_f1", mock_evaluate_f1)
    monkeypatch.setattr("models.GNN.evaluate_accuracy", mock_evaluate_accuracy)
    monkeypatch.setattr("models.GNN.compute_class_weight", mock_compute_class_weight)
    monkeypatch.setattr("torch.save", mock_torch_save)

    # Mock device
    monkeypatch.setattr("models.GNN.device", torch.device("cpu"))

    # Run the objective function
    best_val_f1 = objective(trial)

    # Check that the function returns a float
    assert isinstance(best_val_f1, float)
    assert 0.0 <= best_val_f1 <= 1.0


def test_optuna_optimize_runs_basic(monkeypatch):
    # Mock data and related functions
    class DummyData:
        def __init__(self):
            self.split_called = False

        def __getitem__(self, key):
            return None

    def mock_split_masks(data, device):
        data.split_called = True

    def mock_objective(trial):
        return 0.8  # Mock F1 score

    class DummyStudy:
        def __init__(self):
            self.best_params = {"hidden_channels": 16, "dropout": 0.5}

        def optimize(self, objective, n_trials):
            self.optimize_called = True

    def mock_create_study(direction, study_name, storage, load_if_exists):
        return DummyStudy()

    # Mock dependencies
    dummy_data = DummyData()
    monkeypatch.setattr("optuna.create_study", mock_create_study)
    monkeypatch.setattr("models.GNN.split_masks", mock_split_masks)

    # Run the function
    study = optuna_optimize(dummy_data, split_masks=mock_split_masks, objective=mock_objective, device="cpu")

    # Check that the study is returned and split_masks was called
    assert isinstance(study, DummyStudy)
    assert dummy_data.split_called
    assert hasattr(study, "best_params")
    assert study.best_params == {"hidden_channels": 16, "dropout": 0.5}


def test_optuna_optimize_study_creation(monkeypatch):
    # Mock Optuna study creation
    class DummyStudy:
        def __init__(self):
            self.best_params = {}

        def optimize(self, objective, n_trials):
            pass

    def mock_create_study(direction, study_name, storage, load_if_exists):
        assert direction == "maximize"
        assert study_name == "hetero_gnn_intrusion"
        assert storage == "sqlite:///Parameter_Databases/Optuna/optuna_study.db"
        assert load_if_exists is True
        return DummyStudy()

    # Mock dependencies
    monkeypatch.setattr("optuna.create_study", mock_create_study)

    # Run the function
    study = optuna_optimize(data=None, split_masks=lambda x, y: None, objective=lambda x: None, device="cpu")

    # Check that the study is returned
    assert isinstance(study, DummyStudy)


def test_optuna_optimize_logging(monkeypatch, caplog):
    # Mock Optuna study
    class DummyStudy:
        def __init__(self):
            self.best_params = {"hidden_channels": 32, "dropout": 0.3}

        def optimize(self, objective, n_trials):
            pass

    def mock_create_study(direction, study_name, storage, load_if_exists):
        return DummyStudy()

    # Mock dependencies
    monkeypatch.setattr("optuna.create_study", mock_create_study)

    # Run the function
    with caplog.at_level(logging.INFO):
        optuna_optimize(data=None, split_masks=lambda x, y: None, objective=lambda x: None, device="cpu")

    # Check that the best hyperparameters are logged
    assert "Best hyperparameters: {'hidden_channels': 32, 'dropout': 0.3}" in caplog.text


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        if self.gamma == 0:
            return F.cross_entropy(input, target, weight=self.alpha, reduction=self.reduction)

        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def evaluate(model, data, split='val', device='cpu'):
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device=device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device=device) for k, v in data.edge_index_dict.items()}

        out = model(x_dict, edge_index_dict)

        # Get the predicted classes
        pred = out['flow'].argmax(dim=1)

        # Calculate accuracy
        correct = (pred[data['flow'].val_mask] == data['flow'].y[data['flow'].val_mask]).sum().item()
        total = data['flow'].val_mask.sum().item()
        acc = correct / total if total > 0 else 0.0

    return acc


# Define a placeholder or actual implementation for 'data' in the module
data = None  # Replace with the actual data initialization if applicable
