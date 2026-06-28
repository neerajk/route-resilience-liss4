"""Data package. Only the pure-numpy `indices` are imported eagerly; the
torch/skimage-backed pieces load lazily (PEP 562) so the geo-only preprocessing
(ingest) can import `data.indices` without pulling torch or scikit-image."""
from .indices import ndvi, savi

__all__ = ["ndvi", "savi", "generate_tile", "SyntheticRoadDataset"]


def __getattr__(name):
    if name == "generate_tile":
        from .synthetic import generate_tile
        return generate_tile
    if name == "SyntheticRoadDataset":
        from .dataset import SyntheticRoadDataset
        return SyntheticRoadDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
