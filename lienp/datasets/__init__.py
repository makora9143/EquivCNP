import torch
from torch.utils.data._utils import fetch as tfetch

from .metaset import MetaImageDataset
from .rotmnist import RotationMNIST


__all__ = [
    "MetaImageDataset",
    "RotationMNIST",
]


class _CustomMapDatasetFetcher(tfetch._BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, 'max_total'):
                num_context = torch.randint(3, self.dataset.max_total, size=())
                num_target = torch.randint(3, self.dataset.max_total, size=())
                data = [self.dataset.sample(idx, num_context, num_target) for idx in possibly_batched_index]
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


tfetch._MapDatasetFetcher.fetch = _CustomMapDatasetFetcher.fetch
