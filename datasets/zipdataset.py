from torch.utils.data import Dataset
from typing import Iterable, Tuple, Optional, Callable
import torch

# key = lambda x, y, z: (1, 2, 4)


class ZipDataset(Dataset):
    def __init__(self, *datasets, key):
        super(ZipDataset, self).__init__()
        self.datasets = datasets

        self.idx_targets = []
        _idx = torch.arange(len(self.datasets[0]), dtype=torch.int64)
        # key = key if key is not None else (torch.unique(self.datasets[0].targets), )* len(self.datasets)
        for target in key:
            targets = (
                torch.nonzero(dataset.targets == target[idx]).squeeze()
                for idx, dataset in enumerate(self.datasets)
            )
            self.idx_targets += tuple(zip(*targets))
        self.idx_targets.sort()

    def __len__(self) -> int:
        return len(self.idx_targets)

    def __getitem__(self, index: int):
        return tuple(
            dataset[self.idx_targets[index][idx]]
            for idx, dataset in enumerate(self.datasets)
        )
