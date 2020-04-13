import torch
from torch.utils.data import Dataset


class MetaImageDataset(object):
    """Meta Image Dataset

    Cast a given dataset as a dataset for meta-learning.

    Attributes:
        dataset Dataset: The dataset for meta-learning
        width int: image width
        height int: image height
        max_total int: The max number of total point
        train bool: Shuffle

    """
    def __init__(
            self,
            dataset: Dataset,
            max_total: int = 50,
            train: bool = True
    ):
        self.dataset = dataset
        self.max_total = max_total
        self.train = train

        channel, height, width = dataset[0][0].shape
        self.width = width
        self.height = height
        self.channel = channel

        i = torch.linspace(-self.height / 2., self.height / 2., self.height)
        j = torch.linspace(-self.width / 2., self.width / 2., self.width)
        self.coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __repr__(self):
        return self.__class__.__name__ + ":\n" + self.dataset.__repr__()

    def sample(self, item: int, num_context: int, num_target: int):
        """Sampling points from index data.

        Args:
            item int: item index
            num_context int: The number of context points
            num_target int: The number of target points

        Returns:
            context_coords: Context points coordinates (C, D)
            context_values: Context points values (C, Channel)
            context_masks: Context points mask (C, )
            target_coords: Target points coordinates (T, D)
            target_values: Target points values (T, Channel)
            target_masks: Target points mask (T,)

        """
        if self.train:
            num_total = num_context + num_target
            indices = torch.randint(0, self.width * self.height, size=(num_total,))
        else:
            num_total = self.width * self.height
            indices = torch.arange(num_total)

        row_indices = indices // self.width
        col_indices = indices % self.width

        values, _ = self.dataset[item]

        target_coords = self.coords[row_indices, col_indices]
        target_values = values.permute(1, 2, 0)[row_indices, col_indices]
        target_masks = torch.ones(num_total).bool()

        context_indices = torch.randperm(target_coords.size(0))[:num_context]
        context_coords = target_coords[context_indices]
        context_values = target_values[context_indices]
        context_masks = target_masks[context_indices]
        return ((context_coords, context_values, context_masks),
                (target_coords, target_values, target_masks))

