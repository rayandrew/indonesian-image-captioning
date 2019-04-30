import os

import torch
import torchvision.transforms as transforms

import numpy as np
import h5py

from models import EncoderTagger, EncoderSCN
from datasets import TaggerDataset, SCNDataset

from utils.dataset import create_input_files


# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_folder = './scn_data'  # folder with data files saved by create_input_files.py
base_filename = 'flickr10k_5_cap_per_img_5_min_word_freq'
output_folder = './scn_bottleneck'
# base name shared by data files
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'

# Model parameter
encoded_image_size = 14
semantic_size = 1000
batch_size = 32
workers = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def gen_bottleneck_tagger(semantic_size=1000):
    encoder = EncoderTagger().to(device)
    encoder.fine_tune(False)
    encoder = encoder.to(device)

    data_loader = [
        ('TRAIN', TaggerDataset(data_folder, data_name, 'TRAIN',
                                tag_size=semantic_size,
                                transform=transforms.Compose([normalize]))),

        ('VAL', TaggerDataset(data_folder, data_name, 'VAL',
                              tag_size=semantic_size,
                              transform=transforms.Compose([normalize]))),

        ('TEST', TaggerDataset(data_folder, data_name, 'TEST',
                               tag_size=semantic_size,
                               transform=transforms.Compose([normalize]))),
    ]

    for (split, loader) in data_loader:
        print('Generating bottleneck for {} data'.format(split))
        with h5py.File(os.path.join(output_folder, split + '_TAG_BOTTLENECK_' + base_filename + '.hdf5'), 'w') as h:
            bottlenecks = h.create_dataset(
                'bottlenecks', (len(loader), 2048), dtype='float32')

            for i, data in enumerate(torch.utils.data.DataLoader(loader,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 num_workers=workers,
                                                                 pin_memory=True)):
                imgs = data[0].to(device)
                bottlenecks[i:i + len(imgs)] = encoder(imgs).tolist()

            h.close()
        print('Generating bottleneck for {} data succeed!'.format(split))


def gen_bottleneck_scn():
    encoder = EncoderSCN(encoded_image_size)
    encoder.fine_tune(False)
    encoder = encoder.to(device)

    data_loader = [
        ('TRAIN', SCNDataset(data_folder, data_name, 'TRAIN',
                             transform=transforms.Compose([normalize]))),

        ('VAL', SCNDataset(data_folder, data_name, 'VAL',
                           transform=transforms.Compose([normalize]))),

        ('TEST', SCNDataset(data_folder, data_name, 'TEST',
                            transform=transforms.Compose([normalize]))),
    ]

    for (split, loader) in data_loader:
        print('Generating bottleneck for {} data'.format(split))
        with h5py.File(os.path.join(output_folder, split + '_SCN_BOTTLENECK_' + base_filename + '.hdf5'), 'w') as h:
            bottlenecks = h.create_dataset(
                'bottlenecks', (len(loader), encoded_image_size, encoded_image_size,  2048), dtype='float32')

            for i, data in enumerate(torch.utils.data.DataLoader(loader,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 num_workers=workers,
                                                                 pin_memory=True)):
                imgs = data[0].to(device)
                bottlenecks[i:i + len(imgs)] = encoder(imgs).tolist()

            h.close()
        print('Generating bottleneck for {} data succeed!'.format(split))


if __name__ == '__main__':
    # Create input files (along with word map)
    # print('Gen input files')
    # create_input_files(dataset='flickr10k',
    #                    split_path='./dataset',
    #                    image_folder='./scn_dataset',
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='./scn_data',
    #                    max_len=50)

    # print('Gen bottleneck tagger')
    # gen_bottleneck_tagger()
    # print('Gen bottleneck scn')
    # gen_bottleneck_scn()
    pass
