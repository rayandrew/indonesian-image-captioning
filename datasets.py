import torch
from torch.utils.data import Dataset

import h5py
import json
import os


class CaptionDataset(Dataset):
    r"""A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Arguments
        data_folder (string): folder where data files are stored
        data_name (string): base name of processed datasets
        split (string): split, one of 'TRAIN', 'VAL', or 'TEST'
        transform (torch.Compose): image transform pipeline
    """

    def __init__(self, data_folder, data_name, split, transform=None, cpi=5):
        """
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = cpi if cpi else self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
            j.close()

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            j.close()

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class TagDataset(Dataset):
    r"""A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Arguments
        data_folder: folder where data files are stored
        data_name: base name of processed datasets
        split: split, one of 'TRAIN', 'VAL', or 'TEST'
        transform: image transform pipeline
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load encoded captions (completely into memory)
        self.t = h5py.File(os.path.join(
            data_folder, self.split + '_TAGS_' + data_name + '.hdf5'), 'r')
        self.tags = self.t['tags']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.tags)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        tags = torch.FloatTensor(self.tags[i])

        return img, tags

    def __len__(self):
        return self.dataset_size
