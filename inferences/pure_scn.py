import torch
import torch.nn.functional as F

import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image

from models.encoder.caption import EncoderCaption
from models.encoder.tagger import EncoderTagger

from utils.device import get_device

device = get_device()


def caption_image_beam_search(encoder_img, encoder_tagger, decoder_tagger, decoder_scn, image_path, tag_map, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = encoder_img(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    encoder_tag_out = encoder_tagger(image)
    tag = decoder_tagger(encoder_tag_out).to(device)

    # Flatten tags
    semantic_size = tag.size(1)
    tag = tag.expand(k, semantic_size)

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    # complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder_scn.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder_scn.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        # gating scalar, (s, encoder_dim)
        gate = decoder_scn.sigmoid(decoder_scn.f_beta(h))
        awe = gate * encoder_out

        h, c = decoder_scn.decode_step(
            torch.cat([embeddings, awe], dim=1), tag, (h, c))  # (s, decoder_dim)

        scores = decoder_scn.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            # (s)
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        tag = tag[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    return seq


def main(args):
    print('[(S)emantic (C)ompositional (N)ets] - Generate Caption')

    # Load model
    encoder_img = EncoderCaption()
    encoder_img.fine_tune(False)
    encoder_img = encoder_img.to(device)
    encoder_img.eval()

    encoder_tagger = EncoderTagger()
    encoder_tagger.fine_tune(False)
    encoder_tagger = encoder_tagger.to(device)
    encoder_tagger.eval()

    checkpoint_tagger = torch.load(args.model_tagger)
    decoder_tagger = checkpoint_tagger['decoder']
    decoder_tagger = decoder_tagger.to(device)
    decoder_tagger.eval()

    checkpoint_scn = torch.load(args.model)
    decoder_scn = checkpoint_scn['decoder']
    decoder_scn = decoder_scn.to(device)
    decoder_scn.eval()

    # Load tag map (word2ix)
    with open(args.tag_map, 'r') as j:
        tag_map = json.load(j)
        j.close()
    rev_tag_map = {v: k for k, v in tag_map.items()}  # ix2word

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
        j.close()
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq = caption_image_beam_search(
        encoder_img, encoder_tagger, decoder_tagger, decoder_scn, args.img, tag_map, word_map, args.beam_size)
