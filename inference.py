import argparse
import warnings

import torch

import numpy as np
import json
from scipy.misc import imread, imresize

import torchvision.transforms as transforms

from utils.device import get_device
from utils.loader import load_decoder, att_based_model, scn_based_model
from utils.token import start_token, end_token, unknown_token, padding_token
from utils.url import is_absolute_path, read_image_from_url
from utils.vizualize import visualize_att


warnings.filterwarnings('ignore')

device = get_device()


def read_image(image_path):
    r"""Reads an image and captions it with beam search.

    Arguments
        image_path (String or File Object): path to image
    Return
        Tensor : image tensors  (1, 3, 256, 256)
    """

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)emantic (C)ompositional (N)ets + Attention] - Generate Caption')

    parser.add_argument('--type', '-t', help='model type')
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument(
        '--model_caption', '-mc', default='./pretrained_dict/en/coco/BEST_checkpoint_attention_scn_coco_5_cap_per_img_5_min_word_freq.pth', help='path to pretrained caption model')
    parser.add_argument('--model_tagger', '-mt',
                        default='./pretrained_dict/en/coco/BEST_checkpoint_tagger_coco_5_cap_per_img_5_min_word_freq.pth', help='path to pretrained tagger model')
    parser.add_argument('--tag_map', '-tm', help='path to tag map JSON')
    parser.add_argument('--tag_out_count', '-toc', type=int,
                        default=20, help='count of tag out')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5,
                        type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    need_tag = args.type in scn_based_model
    need_att = args.type in att_based_model

    file_img = args.img

    if is_absolute_path(args.img):
        file_img = read_image_from_url(args.img)

    image = read_image(file_img)

    encoder_tagger = None
    tags = None
    if need_tag:
        print('Load tagger checkpoint..')
        from models.encoders.tagger import EncoderTagger
        tagger_checkpoint = torch.load(
            args.model_tagger, map_location=lambda storage, loc: storage)

        print('Load tagger encoder...')
        encoder_tagger = EncoderTagger()
        encoder_tagger.load_state_dict(tagger_checkpoint['model_state_dict'])
        encoder_tagger = encoder_tagger.to(device)
        encoder_tagger.eval()
        tags = encoder_tagger(image)

    print('Load word map..')
    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    vocab_size = len(word_map)

    # Load word map (word2ix)
    with open(args.tag_map, 'r') as j:
        tag_map = json.load(j)
    rev_tag_map = {v: k for k, v in tag_map.items()}  # ix2word

    caption_checkpoint = torch.load(
        args.model_caption, map_location=lambda storage, loc: storage)

    print('Load caption encoder..')
    from models.encoders.caption import EncoderCaption
    encoder_caption = EncoderCaption()

    encoder_caption.load_state_dict(
        caption_checkpoint['encoder_model_state_dict'])
    encoder_caption = encoder_caption.to(device)
    encoder_caption.eval()

    print('Encoding image...')
    encoder_out = encoder_caption(image)

    print('Load caption decoder..')
    decoder_caption = load_decoder(
        model_type=args.type,
        checkpoint=caption_checkpoint['decoder_model_state_dict'],
        vocab_size=vocab_size)
    decoder_caption.eval()

    print('=========================')

    if need_tag:
        result = decoder_caption.sample(
            args.beam_size, word_map, encoder_out, tags)

        tags = np.asarray(tags.flatten().tolist())
        tag_index = np.argsort(tags)[-args.tag_out_count:]
        print()
        print('Tags defined : ')
        for idx in tag_index:
            print('{} {}'.format(rev_tag_map[idx], tags[idx]))
        print()
    else:
        result = decoder_caption.sample(args.beam_size, word_map, encoder_out)

    print('=========================')

    try:
        seq, alphas = result  # for attention-based model
    except:
        seq = result  # for scn only-based model

    sentences = ' '.join([rev_word_map[ind] for ind in seq if ind not in {
        word_map[start_token], word_map[end_token], word_map[padding_token]}])

    print('Sentences : {}'.format(sentences))
    print()

    if need_att:
        alphas = torch.FloatTensor(alphas)
        # Visualize caption and attention of best sequence
        visualize_att(file_img, seq, alphas, rev_word_map, args.smooth)
