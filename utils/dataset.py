import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

from nltk import pos_tag, tokenize, WordNetLemmatizer

lemma = WordNetLemmatizer()


def get_ground_truth(tags, all_tags, tags_count):
    """Create ground truth array (like one-hot)
    Arguments:
      tags {array} -- image's tags
      all_tags {array} -- tag mapping in number
      tags_count {integer} -- num of tags
    Returns:
      Numpy.Array -- ground truth of image
    """

    ground_truth = np.zeros(tags_count, dtype=np.float32)

    for tag in tags:
        ground_truth[all_tags[tag]] = 1.0

    return ground_truth


def get_tags_en(sentence, tokenize=False):
    tokens = tokenize.word_tokenize(sentence) if tokenize else sentence
    tokens = [lemma.lemmatize(token) for token in tokens]
    tags = pos_tag(tokens)

    # get only nouns
    return [x[0] for x in tags if x[1] in {'NN', 'NNP', 'NNS', 'NNPS'}]


def _filter_data_by_indexes(filenames, data, indexes):
    """This function is used to filter the files and captions based on the indexes
    Arguments:
        filenames {list} -- list of filenames
        captions {list of list} -- list of list of captions
        indexes {list} -- list of indexes
    Returns:
        list, list -- filtered filenames and filtered captions
    """

    filtered_filenames = ()
    filtered_data = ()
    for filename, dat in zip(filenames, data):
        index = filename.split('.')[0]
        if index in indexes:
            filtered_filenames += (filename,)
            filtered_data += ([cap for cap in dat],)
    return filtered_filenames, filtered_data


def load_flickr10k(path_folder):
    r"""Load caption from folder. The folder must has the structure like this:
        /filenames.json -- contain list of filenames
        /captions.json -- contain list of list of caption
        /train.txt -- contain all train indexes
        /val.txt -- contain all validation indexes
        /test.txt -- contain all test indexes

    Arguments:
        path_folder {str} -- string of path folder
    Returns:
        tuple, tuple, tuple -- three tuples of train, val, and test.
        Each tuple has filenames and captions (filename, caption)
    """

    # Load dataset
    with open(os.path.join(path_folder, 'filenames.json'), 'r') as f:
        filenames = json.load(f)
        f.close()
    with open(os.path.join(path_folder, 'tags.json'), 'r') as f:
        tags = json.load(f)
        f.close()
    with open(os.path.join(path_folder, 'captions.json'), 'r') as f:
        captions = json.load(f)
        f.close()

    # Load split indexes
    with open(os.path.join(path_folder, 'train.txt'), 'r') as f:
        train_indexes = [d.rstrip() for d in f.readlines()]
        f.close()
    with open(os.path.join(path_folder, 'val.txt'), 'r') as f:
        val_indexes = [d.rstrip() for d in f.readlines()]
        f.close()
    with open(os.path.join(path_folder, 'test.txt'), 'r') as f:
        test_indexes = [d.rstrip() for d in f.readlines()]
        f.close()

    # Load tag indexes
    with open(os.path.join(path_folder, 'all_tags.txt'), 'r') as f:
        all_tags = [d.rstrip() for d in f.readlines()]
        f.close()

    filenames_train, captions_train = _filter_data_by_indexes(
        filenames=filenames, data=captions, indexes=train_indexes)
    filenames_val, captions_val = _filter_data_by_indexes(
        filenames=filenames, data=captions, indexes=val_indexes)
    filenames_test, captions_test = _filter_data_by_indexes(
        filenames=filenames, data=captions, indexes=test_indexes)

    _, tags_train = _filter_data_by_indexes(
        filenames=filenames, data=tags, indexes=train_indexes)
    _, tags_val = _filter_data_by_indexes(
        filenames=filenames, data=tags, indexes=val_indexes)
    _, tags_test = _filter_data_by_indexes(
        filenames=filenames, data=tags, indexes=test_indexes)

    # Reformat dataset like Karphaty Format
    dataset = {
        'images': [],
        'dataset': 'flickr10k',
        'all_tags': all_tags
    }

    for (filename, caption, tag) in zip(filenames_train, captions_train, tags_train):
        temp = {
            'split': 'train',
            'filename': filename,
            'tags': tag,
        }

        capt = []
        for cap in caption:
            tokens = cap.split()
            capt.append({'tokens': tokens, 'raw': cap})

        temp['sentences'] = capt

        dataset['images'].append(temp)

    for (filename, caption, tag) in zip(filenames_val, captions_val, tags_val):
        temp = {
            'split': 'val',
            'filename': filename,
            'tags': tag,
        }

        capt = []
        for cap in caption:
            tokens = cap.split()
            capt.append({'tokens': tokens, 'raw': cap})

        temp['sentences'] = capt

        dataset['images'].append(temp)

    for (filename, caption, tag) in zip(filenames_test, captions_test, tags_test):
        temp = {
            'split': 'test',
            'filename': filename,
            'tags': tag,
        }

        capt = []
        for cap in caption:
            tokens = cap.split()
            capt.append({'tokens': tokens, 'raw': cap})

        temp['sentences'] = capt

        dataset['images'].append(temp)

    return dataset


def load_tags(labels_file):
    """Get all labels from the dataset
    Arguments:
        labels_file {string} -- path to file that contains all label
    Returns:
        Dictionary -- all labels
    """
    labels = []

    with open(labels_file) as f:
        labels = f.read().splitlines()

        f.close()

    return {labels[i]: i for i in range(len(labels))}


def create_input_files(dataset,
                       split_path,
                       image_folder,
                       captions_per_image,
                       min_word_freq,
                       output_folder,
                       tag_size=1000,
                       max_len=100):
    r"""Creates input files for training, validation, and test data.

    Arguments
        dataset: name of dataset, currently supported {'flick10k', 'coco-id', 'coco', 'flickr30k', 'flickr8k'}
        split_path: path of index, tags, and caption emitted in karpathy json format
        image_folder: folder with downloaded images
        captions_per_image: number of captions to sample per image
        min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
        output_folder: folder to save files
        tag_size: tag vocab count
        max_len: don't sample captions longer than this length
    """

    assert dataset in {'flickr10k', 'coco_id', 'coco', 'flickr30k', 'flickr8k'}
    id_dataset = {'flickr10k', 'coco_id'}

    os.makedirs(output_folder, exist_ok=True)

    if dataset == 'flickr10k':
        data = load_flickr10k(split_path)
    else:
        data = json.load(open(split_path, 'r'))

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_tags = []

    val_image_paths = []
    val_image_captions = []
    val_image_tags = []

    test_image_paths = []
    test_image_captions = []
    test_image_tags = []

    word_freq = Counter()
    all_tags = Counter()

    if dataset not in id_dataset:
        # Iterate to get word_freq and all_tags
        for img in data['images']:
            for c in img['sentences']:
                # Update word frequency
                word_freq.update(c['tokens'])
                all_tags.update(get_tags_en(c['tokens']))

        all_tags = all_tags.most_common(tag_size)
        all_tags = [tag[0] for tag in all_tags]

    for img in data['images']:
        captions = []
        tags = []

        for c in img['sentences']:
            if len(c['tokens']) <= max_len:
                if dataset not in id_dataset:
                    for x in c['tokens']:
                        if x in all_tags:
                            tags.append(x)

                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_captions.append(
                img['tags'] if dataset in id_dataset else tags)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_captions.append(
                img['tags'] if dataset in id_dataset else tags)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_tags.append(
                img['tags'] if dataset in id_dataset else tags)

    # Sanity check
    assert len(train_image_paths) == len(
        train_image_captions) == len(train_image_tags)
    assert len(val_image_paths) == len(
        val_image_captions) == len(val_image_tags)
    assert len(test_image_paths) == len(
        test_image_captions) == len(test_image_tags)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + \
        str(captions_per_image if captions_per_image > -1 else 'all') + '_cap_per_img_' + \
        str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
        j.close()

    # Save tag map to a JSON
    with open(os.path.join(output_folder, 'TAGMAP_' + base_filename + '.json'), 'w') as j:
        tagwordidx = {v: k for k, v in enumerate(
            data['all_tags'] if dataset in id_dataset else all_tags)}
        # idxword = {k: v for k, v in enumerate(data['all_tags'])}
        json.dump(tagwordidx, j)
        j.close()

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imtags, split in [(train_image_paths, train_image_captions, train_image_tags, 'TRAIN'),
                                           (val_image_paths, val_image_captions,
                                            val_image_tags, 'VAL'),
                                           (test_image_paths, test_image_captions, test_image_tags, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'w') as h, \
                h5py.File(os.path.join(output_folder, split + '_TAGS_' + base_filename + '.hdf5'), 'w') as t:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Make a note of the vocab tag vocab defined
            t.attrs['tag_size'] = tag_size

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                'images', (len(impaths), 3, 256, 256), dtype='uint8')

            # Create tags dataset inside HDF5 file to store images
            tags = t.create_dataset(
                'tags', (len(impaths), tag_size), dtype='float32')

            print(
                "\nReading %s images, captions, and tags then storing to file...\n" % split)

            enc_captions = []
            caplens = []
            raw_tags = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i])
                                            for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                # Add tags
                raw_tags.append(imtags[i])
                # Save tags ground truth to HDF5 file
                tags[i] = get_ground_truth(
                    imtags[i], tagwordidx, tag_size)

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] == tags.shape[0]
            assert images.shape[0] * \
                captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)
                j.close()

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)
                j.close()

            # Save tags
            with open(os.path.join(output_folder, split + '_RAWTAGS_' + base_filename + '.json'), 'w') as j:
                json.dump(raw_tags, j)
                j.close()

            t.close()
    #         h.close()
