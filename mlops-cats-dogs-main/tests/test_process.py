from src.data.preprocess import get_transforms

def test_transform_exists():
    transform = get_transforms()
    assert transform is not None
