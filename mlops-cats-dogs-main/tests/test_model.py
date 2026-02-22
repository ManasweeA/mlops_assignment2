import torch
from src.models.model import SimpleCNN

def test_model_output_shape():
    model = SimpleCNN()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 2)
