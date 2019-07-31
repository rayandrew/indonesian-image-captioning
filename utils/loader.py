from utils.device import get_device

device = get_device()

scn_based_model = {'pure_scn', 'attention_scn'}
att_based_model = {'pure_attention', 'attention_scn'}


def load_decoder(model_type,
                 checkpoint,
                 vocab_size,
                 embed_dim=512,
                 attention_dim=512,
                 decoder_dim=512,
                 factored_dim=512,
                 semantic_dim=1000,
                 dropout=0.5):
    r"""Loads a suitable decoder based on its type.
    The default parameters must be same as training params!

    Arguments
        model_type (String): type of model {pure_scn, pure_attention, attention_scn}
        checkpoint (Torch.state_dict): state dict checkpoint
        vocab_size (int): vocabulary size
        embed_dim  (int): word embedding size
        attention_dim (int): attention network size
        decoder_dim (int): decoder size
        factored_dim (int): num of factor in SCN-based model
        semantic_dim (int): size of semantic concept
        dropout (int): dropout rate for decoder
    Return
        nn.Module
    """

    if model_type == 'pure_scn':
        from models.decoders.pure_scn import PureSCN
        decoder_caption = PureSCN(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            factored_dim=factored_dim,
            semantic_dim=semantic_dim,
            vocab_size=vocab_size,
            dropout=dropout)
    elif model_type == 'pure_attention':
        from models.decoders.pure_attention import PureAttention
        decoder_caption = PureAttention(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            dropout=dropout)
    elif model_type == 'attention_scn':
        from models.decoders.attention_scn import AttentionSCN
        decoder_caption = AttentionSCN(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            factored_dim=factored_dim,
            semantic_dim=1000,
            vocab_size=vocab_size,
            dropout=dropout)
    else:
        raise ValueError('Error model type not found!')

    decoder_caption.load_state_dict(checkpoint)
    decoder_caption = decoder_caption.to(device)

    return decoder_caption
