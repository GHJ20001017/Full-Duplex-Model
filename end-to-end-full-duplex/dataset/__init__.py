"""Stage-1 Emilia input pipeline; codecs are injected and frozen by the trainer."""
from .emilia import EmiliaDataset, collate_emilia

__all__ = ["EmiliaDataset", "collate_emilia"]
