import argparse
import os
import json
import time

from tqdm import tqdm

from nlgeval import NLGEval

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets import CaptionDataset

from utils.device import get_device
from utils.loader import load_decoder, att_based_model, scn_based_model
from utils.token import start_token, end_token, padding_token

device = get_device()
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(args):
    r"""
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    # DataLoader
    loader = DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    need_tag = args.type in scn_based_model

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Load tag map (word2ix)
    with open(args.tag_map, 'r') as j:
        tag_map = json.load(j)

    vocab_size = len(word_map)

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

    print('Load caption checkpoint')
    caption_checkpoint = torch.load(
        args.model_caption, map_location=lambda storage, loc: storage)

    print('Load caption encoder..')
    from models.encoders.caption import EncoderCaption
    encoder_caption = EncoderCaption()

    encoder_caption.load_state_dict(
        caption_checkpoint['encoder_model_state_dict'])
    encoder_caption = encoder_caption.to(device)
    encoder_caption.eval()

    print('Load caption decoder..')
    decoder_caption = load_decoder(
        model_type=args.type,
        checkpoint=caption_checkpoint['decoder_model_state_dict'],
        vocab_size=vocab_size)
    decoder_caption.eval()

    print('=========================')

    # Preparing result
    references_temp = list()
    hypotheses = list()

    # For each image
    for i, (image, _, _, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(args.beam_size))):

        k = args.beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder_caption(image)

        # Tag (1, semantic_dim)
        tag_out = encoder_tagger(image)

        if need_tag:
            result = decoder_caption.sample(
                args.beam_size, word_map, encoder_out, tag_out)  # for scn-based model
        else:
            result = decoder_caption.sample(
                args.beam_size, word_map, encoder_out)

        try:
            seq, _ = result  # for attention-based model
        except:
            seq = result  # for scn only-based model

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: ' '.join([rev_word_map[w] for w in c if w not in {word_map[start_token], word_map[end_token], word_map[padding_token]}]),
                img_caps))  # remove <start> and pads
        references_temp.append(img_captions)

        # Hypotheses
        hypotheses.append(' '.join([rev_word_map[w] for w in seq if w not in {
            word_map[start_token], word_map[end_token], word_map[padding_token]}]))

        assert len(references_temp) == len(hypotheses)

    # Calculate Metric scores

    # Modify array so NLGEval can read it
    references = [[] for x in range(len(references_temp[0]))]

    for refs in references_temp:
        for i in range(len(refs)):
            references[i].append(refs[i])

    current_time = round(time.time())

    os.makedirs(os.path.join('evaluation', current_time), exist_ok=True)

    # Creating instance of NLGEval
    n = NLGEval(no_skipthoughts=True, no_glove=True)

    with open(os.path.join('evaluation', current_time, '{}_beam_{}_references.json'.format(args.type, args.beam_size)), 'w') as f:
        json.dump(references, f)
        f.close()

    with open(os.path.join('evaluation', current_time, '{}_beam_{}_hypotheses.json'.format(args.type, args.beam_size)), 'w') as f:
        json.dump(hypotheses, f)
        f.close()

    scores = n.compute_metrics(ref_list=references, hyp_list=hypotheses)

    with open(os.path.join('evaluation', current_time, '{}_beam_{}_scores.json'.format(args.type, args.beam_size)), 'w') as f:
        json.dump(scores, f)
        f.close()

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='[(S)how (A)ttend (T)ell - (S)emantic (C)ompositional (N)etworks] - Eval Caption')

    parser.add_argument('--type', '-t', help='model type')
    parser.add_argument('--model_caption', '-mc',
                        help='path to pretrained caption model')
    parser.add_argument('--model_tagger', '-mt',
                        default='BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar', help='path to pretrained tagger model')
    parser.add_argument('--data_folder', '-df',
                        default='./scn_data', help='data folder')
    parser.add_argument(
        '--data_name', '-dn', default='flickr10k_5_cap_per_img_5_min_word_freq', help='data path')
    parser.add_argument('--tag_map', '-tm', help='path to tag map JSON')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-bs', default=5,
                        type=int, help='beam size')
    args = parser.parse_args()

    score = evaluate(args)

    print("\nScore of {} model @ beam size of {} is {}.\n" %
          (args.type, args.beam_size, score))
