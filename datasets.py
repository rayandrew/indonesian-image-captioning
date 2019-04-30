import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class SCNBottleneckDataset(Dataset):
    """
    A PyTorch SCN Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.dataset_size


class SCNDataset(Dataset):
    """
    A PyTorch SCN Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
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

    def __getitem__(self, i):
        bottleneck = torch.FloatTensor(self.bottlenecks[i])

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        tag = torch.LongTensor(self.tags[i])

        if self.split is 'TRAIN':
            return bottleneck, tag, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // 5) * 5):(((i // 5) * 5) + 5)])
            return bottleneck, tag, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class TaggerBottleneckDataset(Dataset):
    """
    A PyTorch Tagger Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, tag_size, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Tags vocab size
        self.tpi = tag_size

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.tpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.dataset_size


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
