import torch
from torch.utils.data import Dataset

import h5py
import json
import os


class TaggerDataset(Dataset):
    """
    A PyTorch Tagger Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, tag_size):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.b = h5py.File(os.path.join(
            data_folder, self.split + '_TAG_BOTTLENECK_' + data_name + '.hdf5'), 'r')
        self.bottlenecks = self.b['bottlenecks']

        # Tags vocab size
        self.tpi = tag_size

        # Load encoded captions (completely into memory)
        self.t = h5py.File(os.path.join(
            data_folder, self.split + '_TAGS_' + data_name + '.hdf5'), 'r')
        self.tags = self.t['tags']

        # Total number of datapoints
        self.dataset_size = len(self.tags)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        bottleneneck = torch.FloatTensor(self.bottlenecks[i])

        tag = torch.FloatTensor(self.tags[i])

        return bottleneneck, tag

    def __len__(self):
        return self.dataset_size
