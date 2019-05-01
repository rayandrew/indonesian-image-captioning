import time

import neptune

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.encoder.tagger import EncoderTagger
from models.tagger import ImageTagger

from datasets.tagger import TaggerDataset

from utils.device import get_device
from utils.checkpoint import save_tagger_checkpoint_without_encoder
from utils.metric import AverageMeter, binary_accuracy
from utils.optimizer import clip_gradient, adjust_learning_rate

# Data parameters
data_folder = './scn_data'  # folder with data files saved by create_input_files.py
# base name shared by data files
data_name = 'flickr10k_5_cap_per_img_5_min_word_freq'

# Model parameters
bottleneck_size = 2048
semantic_size = 1000
dropout = 0.15

# sets device for model and PyTorch tensors
device = get_device()
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
decoder_lr = 1e-4  # learning rate for encoder if fine-tuning
grad_clip = 5.  # clip gradients at an absolute value of
best_acc = 0.  # Best acc right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none


def main(args):
    """
    Training and validation.
    """

    global best_acc, epochs_since_improvement, checkpoint, start_epoch, data_name

    print('Running on device {}'.format(device))

    print('Initializing neptune-ml')

    neptune.init(project_qualified_name=args.neptune_user + '/' + args.type)

    experiment = neptune.create_experiment(params={
        'epochs': epochs,
        'batch_size': batch_size,
        'bottleneck_size': bottleneck_size,
        'semantic_size': semantic_size,
        'dropout': dropout,
        'device': device,
        'workers': workers,
        'decoder_lr': decoder_lr,
        'grad_clip': grad_clip,
        'adjust_lr_after_epoch': adjust_lr_after_epoch
    })

    experiment.append_tag('resnet152')
    experiment.append_tag('image_tagger')
    experiment.append_tag('indonesian')

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = ImageTagger(bottleneck_size=bottleneck_size,
                              semantic_size=semantic_size,
                              dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_acc = checkpoint['accuracy']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)
    # Loss function
    criterion = nn.BCELoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        TaggerDataset(data_folder, data_name, 'TRAIN',
                      tag_size=semantic_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TaggerDataset(data_folder, data_name, 'VAL',
                      tag_size=semantic_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):
        print('Current epoch {}\n'.format(epoch + 1))

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion=criterion,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        acc = validate(val_loader=val_loader,
                       decoder=decoder,
                       criterion=criterion)

        neptune.send_metric('epoch_acc', epoch, acc)

        # Check if there was an improvement
        is_best = acc.avg > best_acc
        best_acc = max(acc.avg, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        print('Saving checkpoint for epoch {}\n'.format(epoch + 1))

        # Save checkpoint
        save_tagger_checkpoint_without_encoder(data_name, epoch, epochs_since_improvement, decoder,
                                               decoder_optimizer, acc, is_best)

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
    accs = AverageMeter()  # top accuracy

    start = time.time()

    # Batches
    for i, (bottlenecks, tags) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        bottlenecks = bottlenecks.to(device)
        tags = tags.to(device)

        # Forward prop.
        # imgs = encoder(imgs)
        scores = decoder(bottlenecks)
        targets = tags

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        acc = binary_accuracy(scores, targets)
        losses.update(loss.item())
        accs.update(acc)
        batch_time.update(time.time() - start)

        neptune.send_metric('batch_train_accuracy', i, accs.val)
        neptune.send_metric('batch_train_loss', i, losses.val)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses,
                                                                      acc=accs))


def validate(val_loader,
             decoder,
             criterion):
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
    accs = AverageMeter()

    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (bottlenecks, tags) in enumerate(val_loader):

            # Move to device, if available
            bottlenecks = bottlenecks.to(device)
            tags = tags.to(device)

            scores = decoder(bottlenecks)
            targets = tags

            # Calculate loss
            loss = criterion(scores, targets)

            # Keep track of metrics
            losses.update(loss.item())
            top = binary_accuracy(scores, targets)
            accs.update(top)
            batch_time.update(time.time() - start)

            neptune.send_metric('batch_val_accuracy', i, accs.val)
            neptune.send_metric('batch_val_loss', i, losses.val)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, acc=accs))

        print(
            '\n * LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}\n'.format(
                loss=losses,
                acc=accs))

    return accs


if __name__ == '__main__':
    main()
