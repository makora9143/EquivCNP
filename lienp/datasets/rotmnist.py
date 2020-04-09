import os
from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_and_extract_archive, makedir_exist_ok
from torchvision.datasets.vision import VisionDataset

# !wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
# # uncompress the zip file
# !unzip -n mnist_rotation_new.zip -d mnist_rotation_new


class RotationMNIST(VisionDataset):

    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               'You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def mean(self):
        return self.data.mean()

    @property
    def std(self):
        return self.data.std()

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)

        print('Processing...')

        training_set = (
            self.read_image_label_file(os.path.join(self.raw_folder, 'mnist_all_rotation_normalized_float_train_valid.amat'))
        )
        test_set = (
            self.read_image_label_file(os.path.join(self.raw_folder, 'mnist_all_rotation_normalized_float_test.amat'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def read_image_label_file(self, filepath):
        data = np.loadtxt(filepath, delimiter=' ')
        images = torch.as_tensor(data[:, :-1].reshape(-1, 28, 28)).float()
        labels = torch.from_numpy(data[:, -1].astype(np.int64))
        return images, labels

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def default_aug_layers(self):
        return RandomRotateTranslate(0)  # no translation


class RandomRotateTranslate(nn.Module):
    def __init__(self, max_trans=2):
        super().__init__()
        self.max_trans = max_trans

    def forward(self, img):
        if not self.training:
            return img
        bs, _, h, w = img.shape
        angles = torch.rand(bs) * 2 * np.pi
        affineMatrices = torch.zeros(bs, 2, 3)
        affineMatrices[:, 0, 0] = angles.cos()
        affineMatrices[:, 1, 1] = angles.cos()
        affineMatrices[:, 0, 1] = angles.sin()
        affineMatrices[:, 1, 0] = -angles.sin()
        affineMatrices[:, 0, 2] = (2 * torch.rand(bs) - 1) * self.max_trans / w
        affineMatrices[:, 1, 2] = (2 * torch.rand(bs) - 1) * self.max_trans / h
        flowgrid = F.affine_grid(affineMatrices.to(img.device), size=img.shape)
        transformed_img = F.grid_sample(img, flowgrid)
        return transformed_img
