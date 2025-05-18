import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.fixture
def sample_hetero_data():
    data = HeteroData()
    data['host'].x = torch.randn(3, 8)
    data['flow'].x = torch.randn(5, 8)
    data['host', 'connects', 'flow'].edge_index = torch.tensor([[0, 1], [2, 3]])
    data['flow', 'connects', 'host'].edge_index = torch.tensor([[2, 3], [0, 1]])
    data['flow'].y = torch.randint(0, 2, (5,))
    data['flow'].train_mask = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool)
    return data
