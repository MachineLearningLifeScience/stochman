from typing import List

import torch
import numpy as np
from torch.utils.data import TensorDataset


def load_bones_data(train_percentage: float = 0.8, bones: List[int] = None) -> List[TensorDataset]:
    """
    Loads walking.npz, and returns only the bones specified in {bones},
    split into training and testing.
    """
    import pathlib

    data_file = str(pathlib.Path(__file__).parent.absolute()) + "/motion69_06.npz"
    walking = np.load(data_file)
    data = walking["positions"]  # (n_frames)x(n_bones)x3
    radii = walking["radii"]  # (n_bones)

    if bones is not None:
        data = data[:, bones, :]
        radii = radii[bones]

    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)

    train_idx = int(len(data) * train_percentage)
    train_data = data[:train_idx].clone()
    test_data = data[train_idx:].clone()

    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    return train_dataset, test_dataset, radii
