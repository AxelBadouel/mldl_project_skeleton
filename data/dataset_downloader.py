"""
    In this file, I put the function meant to download the datasets and add them in the dataset folder at the path:
     /content/mldl_project_skeleton/dataset

    This path is the path to the dataset IN Colab **NOT** on the local machine.
    This is because running this function and therefore downloading the dataset makes more sense in collab as the GPUs are all there.
"""
import torch
from torchvision import datasets, models, transforms

def download_dataset():
    image_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]   
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "/content/mldl_project_skeleton/dataset",
            train=True,
            download=True,
            transform=image_transforms
        ),
        batch_size=64,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "/content/mldl_project_skeleton/dataset",
            train=False,
            transform=image_transforms
        ),
        batch_size=1000,
        shuffle=True,
    )
    # Retrieve the image size and the number of color channels
    x, yy = next(iter(train_loader))

    n_channels = x.shape[1]
    input_size_w = x.shape[2]
    input_size_h = x.shape[3]
    input_size = input_size_w * input_size_h

    # Specify the number of classes in CIFAR10
    output_size = yy.max().item() + 1  # there are 10 classes
    output_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')