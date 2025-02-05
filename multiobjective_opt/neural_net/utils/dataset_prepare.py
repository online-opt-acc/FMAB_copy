from abc import ABC, abstractmethod
from itertools import cycle

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LoaderCycleHandler:
    def __init__(self, dataloader: DataLoader, cycled=False, iterator_steps=None):
        self.cycled = cycled
        if cycled:
            self.iterator_steps = iterator_steps
            self.dataloader = cycle(dataloader)
        else:
            self.iterator_steps = len(dataloader)
            self.dataloader = dataloader

    def get_iterator(
        self,
    ):
        if self.cycled:

            def _iter():
                for _, elems in zip(range(self.iterator_steps), self.dataloader):
                    yield elems

            return _iter()

        else:
            return self.dataloader


class ClassificationDatasetsHandlerBase(ABC):
    def __init__(self, cycled=False, iterator_steps=None):
        self.cycled = cycled
        self.iterator_steps = iterator_steps

    @property
    @abstractmethod
    def item_shape(
        self,
    ):
        """
        returns shape of item for this dataset
        """

    @property
    @abstractmethod
    def num_classes(
        self,
    ):
        """
        return number of classes
        """

    @abstractmethod
    def _load_dataset(self, batch_size=128):
        """
        there you should load and return train and test datasets
        """

    def load_dataset(self, batch_size=128):
        """
        loads and returns train and test dataset
        """
        train_dataset, test_dataset = self._load_dataset(batch_size)
        train_dataset = LoaderCycleHandler(
            train_dataset, self.cycled, self.iterator_steps
        )
        return train_dataset, test_dataset


class MNISTHandler(ClassificationDatasetsHandlerBase):
    @property
    def item_shape(self):
        return (1, 28, 28)

    @property
    def num_classes(
        self,
    ):
        return 10

    def _load_dataset(self, batch_size=128):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # load datasets
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # make_dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader


class CIFAR10Handler(ClassificationDatasetsHandlerBase):
    @property
    def item_shape(self):
        return (3, 32, 32)

    @property
    def num_classes(
        self,
    ):
        return 10

    def _load_dataset(self, batch_size=128):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # load_dataset
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # make_dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # CIFAR-10 classes
        # classes = (
        #     "plane",
        #     "car",
        #     "bird",
        #     "cat",
        #     "deer",
        #     "dog",
        #     "frog",
        #     "horse",
        #     "ship",
        #     "truck",
        # )
        return train_loader, test_loader  # , classes
