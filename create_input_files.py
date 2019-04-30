import os

import torch
import torchvision.transforms as transforms

import numpy as np
import h5py

from models import EncoderTagger, EncoderSCN, ImageTagger
from datasets import TaggerDataset, SCNDataset, TaggerBottleneckDataset, SCNBottleneckDataset

from utils.dataset import create_input_files


# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_folder = './scn_data'  # folder with data files saved by create_input_files.py
base_filename = 'flickr10k_5_cap_per_img_5_min_word_freq'
# base name shared by data files
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'

# Model parameter
bottleneck_size = 2048
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
    encoder.eval()

    data_loader = [
        ('TRAIN', TaggerBottleneckDataset(data_folder, data_name, 'TRAIN',
                                          tag_size=semantic_size,
                                          transform=transforms.Compose([normalize]))),

        ('VAL', TaggerBottleneckDataset(data_folder, data_name, 'VAL',
                                        tag_size=semantic_size,
                                        transform=transforms.Compose([normalize]))),

        ('TEST', TaggerBottleneckDataset(data_folder, data_name, 'TEST',
                                         tag_size=semantic_size,
                                         transform=transforms.Compose([normalize]))),
    ]

    for (split, loader) in data_loader:
        print('Generating bottleneck for {} data'.format(split))
        with h5py.File(os.path.join(data_folder, split + '_TAG_BOTTLENECK_' + base_filename + '.hdf5'), 'w') as h:
            bottlenecks = h.create_dataset(
                'bottlenecks', (len(loader), bottleneck_size), dtype='float32')

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
    encoder.eval()

    data_loader = [
        ('TRAIN', SCNBottleneckDataset(data_folder, data_name, 'TRAIN',
                                       transform=transforms.Compose([normalize]))),

        ('VAL', SCNBottleneckDataset(data_folder, data_name, 'VAL',
                                     transform=transforms.Compose([normalize]))),

        ('TEST', SCNBottleneckDataset(data_folder, data_name, 'TEST',
                                      transform=transforms.Compose([normalize]))),
    ]

    for (split, loader) in data_loader:
        print('Generating bottleneck for {} data'.format(split))
        with h5py.File(os.path.join(data_folder, split + '_SCN_BOTTLENECK_' + base_filename + '.hdf5'), 'w') as h:
            bottlenecks = h.create_dataset(
                'bottlenecks', (len(loader), encoded_image_size, encoded_image_size,  bottleneck_size), dtype='float32')

            for i, data in enumerate(torch.utils.data.DataLoader(loader,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 num_workers=workers,
                                                                 pin_memory=True)):
                imgs = data[0].to(device)
                bottlenecks[i:i + len(imgs)] = encoder(imgs).tolist()

            h.close()
        print('Generating bottleneck for {} data succeed!'.format(split))


def gen_tag(model='./BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar'):
    checkpoint = torch.load(model)

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

    data_loader = [
        ('TRAIN', TaggerDataset(data_folder, data_name, 'TRAIN',
                                tag_size=semantic_size)),

        ('VAL', TaggerDataset(data_folder, data_name, 'VAL',
                              tag_size=semantic_size)),

        ('TEST', TaggerDataset(data_folder, data_name, 'TEST',
                               tag_size=semantic_size)),
    ]

    for (split, loader) in data_loader:
        print('Generating bottleneck for {} data'.format(split))
        with h5py.File(os.path.join(data_folder, split + '_SCN_TAG_PROBS_' + base_filename + '.hdf5'), 'w') as h:
            probs = h.create_dataset(
                'probs', (len(loader), semantic_size), dtype='float32')

            for i, data in enumerate(torch.utils.data.DataLoader(loader,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 num_workers=workers,
                                                                 pin_memory=True)):
                imgs = data[0].to(device)
                probs[i:i + len(imgs)] = decoder(imgs).tolist()

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
    #                    data_folder='./scn_data',
    #                    max_len=50)

    # print('Gen bottleneck tagger')
    # gen_bottleneck_tagger()
    # print('Gen bottleneck scn')
    # gen_bottleneck_scn()
    gen_tag()
