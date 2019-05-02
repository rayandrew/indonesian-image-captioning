import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import json

from datasets.scn import SCNDataset
from utils.device import get_device

from nlgeval import NLGEval

from tqdm import tqdm

# sets device for model and PyTorch tensors
device = get_device()
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True


def evaluate(args):
    """
    Evaluation

    :param args: args to generate captions for evaluation
    :return: BLEU-4 score
    """
    # Compute metrics
    n = NLGEval(no_skipthoughts=True)

    # Load model
    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

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
    vocab_size = len(word_map)

    # DataLoader
    loader = DataLoader(
        SCNDataset(args.data_folder, args.data_name, 'TEST', pure_scn=True),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references_temp = list()
    hypotheses = list()

    # For each image
    for i, (bottleneck, tags, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(args.beam_size))):

        k = args.beam_size

        # Move to GPU device, if available
        encoder_out = bottleneck.to(device)  # (1, 3, 256, 256)
        tags = tags.to(device)  # (1, 1000)

        # Encode
        # (1, encoder_dim)
        encoder_dim = encoder_out.size(1)

        # Flatten encoding
        # (1, encoder_dim)
        encoder_out = encoder_out.view(1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, encoder_dim)

        # (k, 1000)
        semantic_size = tags.size(1)
        tags = tags.expand(k, semantic_size)

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

            # gating scalar, (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * encoder_out

            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), tags, (h, c))  # (s, decoder_dim)

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
            tags = tags[prev_word_inds[incomplete_inds]]
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
            map(lambda c: ' '.join([rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]),
                img_caps))  # remove <start> and pads
        references_temp.append(img_captions)

        # Hypotheses
        hypotheses.append(' '.join([rev_word_map[w] for w in seq if w not in {
                          word_map['<start>'], word_map['<end>'], word_map['<pad>']}]))

        assert len(references_temp) == len(hypotheses)

    references = [[] for x in range(len(references_temp[0]))]

    for refs in references_temp:
        for i in range(len(refs)):
            references[i].append(refs[i])

    scores = n.compute_metrics(ref_list=references, hyp_list=hypotheses)

    return scores
