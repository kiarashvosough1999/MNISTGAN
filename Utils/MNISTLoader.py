import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


class MNISTLoader:

    def __init__(self):
        self._batch_size = 512

    def load_MNIST(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_data = datasets.FashionMNIST(
            root='input/data',
            train=True,
            download=True,
            transform=transform
        )
        return DataLoader(train_data, batch_size=self._batch_size, shuffle=True), train_data
