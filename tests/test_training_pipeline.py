import torch
from torch_geometric.data import HeteroData
from src.models.Script_ver import train, evaluate_model


def test_training_step_runs():
    model = torch.nn.Module()  # Mock model or import actual model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fake batch
    batch = HeteroData()
    batch['host'].x = torch.randn(4, 8)
    batch['flow'].x = torch.randn(6, 8)
    batch['host', 'connects', 'flow'].edge_index = torch.tensor([[0, 1], [2, 3]])
    batch['flow', 'connects', 'host'].edge_index = torch.tensor([[2, 3], [0, 1]])
    batch['flow'].y = torch.randint(0, 2, (6,))
    batch['flow'].train_mask = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)

    # Dummy train step
    loss = train(model, optimizer, batch, torch.nn.CrossEntropyLoss(), device="cpu")
    assert loss >= 0
