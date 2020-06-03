






from torchvision.datasets.vision import VisionDataset

class affNIST(VisionDataset):
    """affNIST

    Affine MNIST

    http://www.cs.toronto.edu/~tijmen/affNIST/

    0: the affine transformation matrix
    1: the affine transformation inverse matrix
    2: Image Data
    3: The index (1-based) of the data case in the affNIST dataset
    4: One-hot vector
    5: Label
    6: Transformation
    7: The index (1-based) of the original image

    """

    resources = [
        "http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_batches.zip",

    ]