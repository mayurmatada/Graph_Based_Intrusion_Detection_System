import torch
import types
import polars as pl
import numpy as np
from models.GNN import torch_config, process_and_load_data, split_masks, train, evaluate_f1  # type: ignore


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


def test_train_loss_decreases(monkeypatch):
    # Simple model and data where loss can decrease
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x_dict, edge_index_dict):
            return {'flow': self.lin(x_dict['flow'])}

    class DummyData:
        def __init__(self, device):
            self.x_dict = {'flow': torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)}
            self.edge_index_dict = {}
            self.flow = types.SimpleNamespace()
            self.flow.train_mask = torch.tensor([True, True], device=device)
            self.flow.y = torch.tensor([0, 1], device=device)

    model = DummyModel()
    device = next(model.parameters()).device
    model = model.to(device)
    data = DummyData(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    # Run multiple steps and check loss decreases
    prev_loss = None
    for _ in range(5):
        loss = train(model, data, optimizer, criterion, scaler)
        if prev_loss is not None:
            # Allow for small fluctuations, but expect general decrease
            assert loss <= prev_loss + 1e-3 or abs(loss - prev_loss) < 1e-2
        prev_loss = loss
