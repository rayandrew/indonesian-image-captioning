import torch


def save_checkpoint(model_name, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    r"""Saves model checkpoint.

    Arguments
        model_name (String): base name of caption model
        data_name (String): base name of processed dataset
        epoch (Int): epoch number
        epochs_since_improvement (int): number of epochs since last improvement in BLEU-4 score
        encoder (torch.nn.Module): encoder model
        decoder (torch.nn.Module): decoder model
        encoder_optimizer (Optimizer): optimizer to update encoder's weights, if fine-tuning
        decoder_optimizer (Optimizer): optimizer to update decoder's weights
        bleu4 (Float): validation BLEU-4 score for this epoch
        is_best (boolean): is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + model_name + '_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_tagger_checkpoint(data_name,
                           epoch, epochs_since_improvement,
                           encoder,
                           encoder_optimizer,
                           accuracy,
                           is_best):
    r"""Saves model tagger checkpoint.

    Arguments
        data_name (String): base name of processed dataset
        epoch (Int): epoch number
        epochs_since_improvement (int): number of epochs since last improvement in BLEU-4 score
        encoder (torch.nn.Module): encoder model
        encoder_optimizer (Optimizer): optimizer to update encoder's weights, if fine-tuning
        bleu4 (Float): validation BLEU-4 score for this epoch
        is_best (boolean): is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'accuracy': accuracy,
             'encoder': encoder,
             'encoder_optimizer': encoder_optimizer}
    filename = 'checkpoint_tagger_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)
