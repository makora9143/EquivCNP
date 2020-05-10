import os
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url


class ClockDigit(VisionDataset):
    """
    """
    resources = [
        'https://www.ht.sfc.keio.ac.jp/~makora/clockdigits/clock-digits.pth'
    ]
    data_file = 'clock-digits.pth'

    def __init__(self, root, download=False, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download")

        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, self.data_file))

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_url(url, root=self.processed_folder, filename=filename)
        print('Done!')
        return
