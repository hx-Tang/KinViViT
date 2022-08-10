import torch
from torch.utils.data import Dataset
import numpy as np


class CIFAR100TwoStreamDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, anchor_transform, posneg_transform):
        # split by some thresholds here 80% anchors, 20% for posnegs
        lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
        self.anchors, self.posnegs = torch.utils.data.random_split(dataset, lengths)

        self.anchor_transform = anchor_transform
        self.posneg_transform = posneg_transform

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor, target = self.anchors[index]
        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)

        # now pair this up with an image from the same class in the second stream
        A = np.where(np.array(self.posnegs.dataset.targets) == target)[0]
        posneg_idx = np.random.choice(A[np.in1d(A, self.posnegs.indices)])
        posneg, target = self.posnegs[
            np.where(self.posnegs.indices == posneg_idx)[0][0]
        ]

        if self.posneg_transform is not None:
            posneg = self.posneg_transform(posneg)
        return anchor, posneg, target