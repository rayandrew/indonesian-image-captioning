import torch
import torch.nn.functional as F

import numpy as np
import json
import torchvision.transforms as transforms

import argparse

from scipy.misc import imread, imresize

from utils.url import is_absolute_path, read_image_from_url
from utils.vizualize import visualize_att

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, encoder_tagger, decoder, image_path, word_map, beam_size=3):
    r"""Reads an image and captions it with beam search.

    Arguments
        encoder (nn.Module): encoder model
        encoder_tagger (nn.Module): encoder tagger model
        decoder (nn.Module): decoder model
        image_path (String or File Object): path to image
        word_map (Dictionary): word map
        beam_size (int, optional): number of sequences to consider at each decode-step
    Return
        String : caption
        Float  : weights for visualization
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
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    tag_out = encoder_tagger(image)
    tag_size = tag_out.size(1)

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    temp_tag_out = tag_out.expand(k, tag_size)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
        device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)

        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), temp_tag_out, (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
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
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        temp_tag_out = temp_tag_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas, tag_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[Indonesian Image Captioning] -- (S)how (A)ttend and (T)ell + (S)emantic (C)ompositional (N)etworks -- Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument(
        '--model', '-m', default='./pretrained/BEST_checkpoint_attention_scn_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to model')
    parser.add_argument(
        '--tagger_model', '-mt', default='./pretrained/BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to model')
    parser.add_argument(
        '--word_map', '-wm', default='./scn_data/WORDMAP_flickr10k_5_cap_per_img_5_min_word_freq.json', help='path to word map JSON')
    parser.add_argument(
        '--tag_map', '-tm', default='./scn_data/TAGMAP_flickr10k_5_cap_per_img_5_min_word_freq.json', help='path to tag map JSON')
    parser.add_argument(
        '--tag_out_count', '-toc', default=20, type=int, help='count tag output')
    parser.add_argument('--beam_size', '-b', default=5,
                        type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    file_img = args.img

    if is_absolute_path(args.img):
        file_img = read_image_from_url(args.img)

    # Load model
    checkpoint = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    tagger_checkpoint = torch.load(
        args.tagger_model, map_location=lambda storage, loc: storage)

    encoder_tagger = tagger_checkpoint['encoder']
    encoder_tagger = encoder_tagger.to(device)
    encoder_tagger.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Load word map (word2ix)
    with open(args.tag_map, 'r') as j:
        tag_map = json.load(j)
    rev_tag_map = {v: k for k, v in tag_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas, tags = caption_image_beam_search(
        encoder, encoder_tagger, decoder, file_img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    tags = np.asarray(tags.flatten().tolist())
    tag_index = np.argsort(tags)[-args.tag_out_count:]

    print()
    print('Tags defined')
    for idx in tag_index:
        print('{} {}'.format(rev_tag_map[idx], tags[idx]))
    print()

    # Visualize caption and attention of best sequence
    visualize_att(file_img, seq, alphas, rev_word_map, args.smooth)
