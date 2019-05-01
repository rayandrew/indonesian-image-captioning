import time
import os
import json

import neptune

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.decoder.pure_attention import PureAttention

from datasets.scn import SCNDataset

from utils.checkpoint import save_caption_checkpoint_without_encoder
from utils.metric import AverageMeter, accuracy
from utils.optimizer import clip_gradient, adjust_learning_rate

from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = './scn_data'  # folder with data files saved by create_input_files.py
# base name shared by data files
topology_name = 'pure_attention'
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN

dropout = 0.5
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 8
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
adjust_lr_after_epoch = 4
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py

decoder_lr = 4e-4  # learning rate for decoder scn

grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder_scn = False  # fine-tune encoder scn?
fine_tune_encoder_tagger = False  # fine-tune encoder tagger?
checkpoint = None  # path to checkpoint, None if none


def main(args):
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder_scn, fine_tune_encoder_tagger, data_name, word_map

    print('Running on device {}'.format(device))

    print('Initializing neptune-ml')

    neptune.init(api_token=args.neptune_key,
                 project_qualified_name=args.neptune_user + '/' + args.type)

    experiment = neptune.create_experiment(name=args.type,
                                           params={
                                               'epochs': epochs,
                                               'batch_size': batch_size,
                                               'emb_dim': emb_dim,
                                               'attention_dim': attention_dim,
                                               'decoder_dim': decoder_dim,
                                               'workers': workers,
                                               'decoder_lr': decoder_lr,
                                               'alpha_c': alpha_c,
                                               'grad_clip': grad_clip,
                                               'adjust_lr_after_epoch': adjust_lr_after_epoch
                                           })

    experiment.append_tag('resnet152')
    experiment.append_tag('pure_attention')
    experiment.append_tag('indonesian')
    experiment.append_tag('image_caption')

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = PureAttention(attention_dim=attention_dim,
                                embed_dim=emb_dim,
                                decoder_dim=decoder_dim,
                                vocab_size=len(word_map),
                                dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        SCNDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        SCNDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('Start Training')

    # Epochs
    for epoch in range(start_epoch, epochs):
        print('Current epoch {}\n'.format(epoch + 1))

        # Decay learning rate if there is no improvement for [adjust_lr_after_epoch] consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % adjust_lr_after_epoch == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion=criterion,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion=criterion)

        neptune.send_metric('recent_bleu4', epoch, recent_bleu4)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        print('Saving checkpoint for epoch {}\n'.format(epoch + 1))

        # Save checkpoint
        save_caption_checkpoint_without_encoder(topology_name,
                                                data_name,
                                                epoch,
                                                epochs_since_improvement,
                                                decoder,
                                                decoder_optimizer,
                                                recent_bleu4,
                                                is_best)

    neptune.stop()


def train(train_loader,
          decoder,
          criterion,
          decoder_optimizer,
          epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion: loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (bottlenecks, _, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        bottlenecks = bottlenecks.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            bottlenecks, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        neptune.send_metric('batch_train_accuracy', i, top5accs.val)
        neptune.send_metric('batch_train_loss', i, losses.val)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (bottlenecks, _, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            bottlenecks = bottlenecks.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                bottlenecks, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            neptune.send_metric('batch_val_accuracy', i, top5accs.val)
            neptune.send_metric('batch_val_loss', i, losses.val)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4
