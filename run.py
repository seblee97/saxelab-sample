import argparse

import model
import runner
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)


def get_dataloaders(batch_size):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


if __name__ == "__main__":

    args = parser.parse_args()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    net = model.Net().to(device)
    train_dataloader, test_dataloader = get_dataloaders(args.batch_size)

    r = runner.Runner(
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        lr=args.learning_rate,
    )

    r.train(num_epochs=args.num_epochs)
