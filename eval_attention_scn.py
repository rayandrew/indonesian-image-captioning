import os
import json

from tqdm import tqdm

from nlgeval import NLGEval

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import CaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F

# Parameters
# folder with data files saved by create_input_files.py
data_folder = './scn_data'
# base name shared by data files
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'
# model checkpoint
checkpoint = './BEST_checkpoint_attention_scn_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar'
tagger_checkpoint = './BEST_checkpoint_tagger_flickr10k_5_cap_per_img_5_min_word_freq.pth.tar'
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = './scn_data/WORDMAP_flickr10k_5_cap_per_img_5_min_word_freq.json'
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

tagger_checkpoint = torch.load(tagger_checkpoint)
encoder_tagger = tagger_checkpoint['encoder']
encoder_tagger = encoder_tagger.to(device)
encoder_tagger.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references_temp = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(image)
        # enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Tag
        # (1, semantic_dim)
        tag_out = encoder_tagger(image)
        semantic_dim = tag_out.size(1)

        # Flatten encoding
        # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        tag_out = tag_out.expand(k, semantic_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            # (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h)

            # gating scalar, (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe

            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), tag_out,  (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
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
            tag_out = tag_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references_temp.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {
                          word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references_temp) == len(hypotheses)

    # Calculate Metric scores

    # Modify array so NLGEval can read it
    references = [[] for x in range(len(references_temp[0]))]

    for refs in references_temp:
        for i in range(len(refs)):
            references[i].append(refs[i])

    os.makedirs("./evaluation", exist_ok=True)

    # Creating instance of NLGEval
    n = NLGEval(no_skipthoughts=True)

    with open('./evaluation/attention_scn_beam_{}_references.json'.format(beam_size), 'w') as f:
        json.dump(references, f)
        f.close()

    with open('./evaluation/attention_scn_beam_{}_hypotheses.json'.format(beam_size), 'w') as f:
        json.dump(hypotheses, f)
        f.close()

    scores = n.compute_metrics(ref_list=references, hyp_list=hypotheses)

    with open('./evaluation/attention_scn_beam_{}_scores.json'.format(beam_size), 'w') as f:
        json.dump(scores, f)
        f.close()

    return scores


if __name__ == '__main__':
    beam_size = 5
    print("\nScores of Attention + SCN @ beam size of {} is {}.".format(
        beam_size, evaluate(beam_size)))