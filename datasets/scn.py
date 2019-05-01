import torch
from torch.utils.data import Dataset

import h5py
import os
import json


class SCNDataset(Dataset):
    """
    A PyTorch SCN Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, cpi=5):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.b = h5py.File(os.path.join(
            data_folder, self.split + '_SCN_BOTTLENECK_' + data_name + '.hdf5'), 'r')
        self.bottlenecks = self.b['bottlenecks']

        # Open hdf5 file where tags are stored
        self.t = h5py.File(os.path.join(
            data_folder, self.split + '_SCN_TAG_PROBS_' + data_name + '.hdf5'), 'r')
        self.tags = self.t['probs']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
            j.close()

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            j.close()

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        self.cpi = 5

    def __getitem__(self, i):
        bottleneck = torch.FloatTensor(self.bottlenecks[i // self.cpi])

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        tag = torch.FloatTensor(self.tags[i // self.cpi])

        if self.split is 'TRAIN':
            return bottleneck, tag, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return bottleneck, tag, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
