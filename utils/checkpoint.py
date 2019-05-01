import torch


def save_tagger_checkpoint(data_name,
                           epoch, epochs_since_improvement,
                           encoder,
                           decoder,
                           encoder_optimizer,
                           decoder_optimizer,
                           accuracy,
                           is_best):
    """
    Saves model tagger checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param accuracy: accuracy for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'accuracy': accuracy,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_tagger_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_tagger_checkpoint_without_encoder(data_name,
                                           epoch, epochs_since_improvement,
                                           decoder,
                                           decoder_optimizer,
                                           accuracy,
                                           is_best):
    """
    Saves model tagger checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param accuracy: accuracy for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'accuracy': accuracy,
             'decoder': decoder,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_tagger_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_caption_checkpoint(data_name,
                            epoch,
                            epochs_since_improvement,
                            encoder_scn,
                            encoder_tagger,
                            decoder,
                            encoder_scn_optimizer,
                            encoder_tagger_optimizer,
                            decoder_optimizer,
                            bleu4, is_best):
    """
    Saves model scn checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder_scn: encoder scn model
    :param encoder_tagger: encoder tagger model
    :param decoder: decoder model
    :param encoder_scn_optimizer: optimizer to update encoder scn's weights, if fine-tuning
    :param encoder_tagger_optimizer: optimizer to update encoder tagger's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder_scn': encoder_scn,
             'encodr_tagger': encoder_tagger,
             'decoder': decoder,
             'encoder_scn_optimizer': encoder_scn_optimizer,
             'encoder_tagger_optimizer': encoder_tagger_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_scn_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_caption_checkpoint_without_encoder(topology_name,
                                            data_name,
                                            epoch,
                                            epochs_since_improvement,
                                            # encoder_scn,
                                            # encoder_tagger,
                                            decoder,
                                            # encoder_scn_optimizer,
                                            # encoder_tagger_optimizer,
                                            decoder_optimizer,
                                            bleu4, is_best):
    """
    Saves model scn checkpoint.

    :param topology_name: topology name of model
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder_scn: encoder scn model
    :param encoder_tagger: encoder tagger model
    :param decoder: decoder model
    :param encoder_scn_optimizer: optimizer to update encoder scn's weights, if fine-tuning
    :param encoder_tagger_optimizer: optimizer to update encoder tagger's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             #  'encoder_scn': encoder_scn,
             #  'encodr_tagger': encoder_tagger,
             'decoder': decoder,
             #  'encoder_scn_optimizer': encoder_scn_optimizer,
             #  'encoder_tagger_optimizer': encoder_tagger_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + topology_name + '_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)
