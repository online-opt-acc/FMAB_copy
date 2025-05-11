from abc import ABC, abstractmethod
from itertools import cycle

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision
import numpy as np

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


class ClassifDatasetHandlerBase(ABC):
    def __init__(self, cycled=False, iterator_steps=None, root="./data"):
        self.cycled = cycled
        self.iterator_steps = iterator_steps
        self.root = root

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
        train_dataset, val_dataset, test_dataset = self._load_dataset(batch_size)
        train_dataset = LoaderCycleHandler(
            train_dataset, self.cycled, self.iterator_steps
        )
        return train_dataset, val_dataset, test_dataset


class MNISTHandler(ClassifDatasetHandlerBase):
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
            root=self.root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=self.root, train=False, download=True, transform=transform
        )

        # make_dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader


class CIFAR10Handler(ClassifDatasetHandlerBase):
    @property
    def item_shape(self):
        return (3, 32, 32)

    @property
    def num_classes(
        self,
    ):
        return 10

    def _load_dataset(self, batch_size=128, valid_size = 0.1):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # load_dataset
        train_dataset = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=transform
        )

        # partitioner
        train_length = len(train_dataset)
        indices=list(range(train_length))
        split = int(np.floor(valid_size * train_length))
        
        np.random.shuffle(indices)

        train_idx=indices[split:]
        valid_idx=indices[:split]

        train_sampler=SubsetRandomSampler(train_idx)
        validation_sampler=SubsetRandomSampler(valid_idx)
        # partitioner

        # make_dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader,val_loader, test_loader  # , classes


class CIFAR100Handler(ClassifDatasetHandlerBase):
    @property
    def item_shape(self):
        return (3, 32, 32)
    @property
    def num_classes(
        self,
    ):
        return 100
    
    def _load_dataset(self, batch_size=128, valid_size = 0.1):
        # Define data preprocessing and augmentation
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
                transforms.RandomCrop(32, padding=4),  # Randomly crop images
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize with CIFAR-100 mean and std
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize with CIFAR-100 mean and std
            ])

        # Load CIFAR-100 dataset
        train_dataset = torchvision.datasets.CIFAR100(
                root=self.root,  # Path to store the dataset
                train=True,  # Load training data
                download=True,  # Download if not already present
                transform=transform_train  # Apply training transformations
            )

        test_dataset = torchvision.datasets.CIFAR100(
                root='./data',  # Path to store the dataset
                train=False,  # Load test data
                download=True,  # Download if not already present
                transform=transform_test  # Apply test transformations
            )


        # partitioner
        train_length = len(train_dataset)
        indices=list(range(train_length))
        split = int(np.floor(valid_size * train_length))
        
        np.random.shuffle(indices)

        train_idx=indices[split:]
        valid_idx=indices[:split]

        train_sampler=SubsetRandomSampler(train_idx)
        validation_sampler=SubsetRandomSampler(valid_idx)
        # partitioner

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)

        test_loader = DataLoader(
                test_dataset,  # Test dataset
                batch_size=batch_size,  # Batch size
                shuffle=False,  # No need to shuffle test data
                num_workers=2  # Number of subprocesses for data loading
            )
        return train_loader, val_loader, test_loader