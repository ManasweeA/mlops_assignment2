import pytest
from src.data_prep import preprocess_data
from PIL import Image

def test_preprocess():
    train_ds, _, _ = preprocess_data()
    img, label = train_ds[0]
    assert img.shape == (3, 224, 224)  # Normalized tensor
