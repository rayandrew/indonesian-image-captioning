import torch
from torch import nn
import torchvision

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_tensor1d(tensor, split):
    return [
        tensor[:split],
        tensor[split: split * 2],
        tensor[split * 2: split * 3],
        tensor[split * 3:],
    ]


def split_tensor2d(tensor, split, front=False):
    if front:
        return [
            tensor[:split, :],
            tensor[split: split * 2, :],
            tensor[split * 2: split * 3, :],
            tensor[split * 3:, :],
        ]

    return [
        tensor[:, :split],
        tensor[:, split: split * 2],
        tensor[:, split * 2: split * 3],
        tensor[:, split * 3:],
    ]


class EncoderCaption(nn.Module):
    r"""Image Encoder using ResNet152.

    Arguments:
        encoded_image_size (int, optional): size of encoded image
    """

    def __init__(self, encoded_image_size=14):
        super(EncoderCaption, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet152(
            pretrained=True)  # pretrained ImageNet ResNet-152

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        r"""Forward propagation.

        Arguments 
            images (torch.Tensor): images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        Returns:
            torch.Tensor: Image Tensor with dimensions (batch_size, encoded_image_size, encoded_image_size, 2048)
        """
        out = self.resnet(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        r"""Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Arguments
            fine_tune (boolean): Allow fine tuning?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class EncoderTagger(nn.Module):
    r"""Tagger Encoder extends ResNet152 Model.

    Arguments:
        semantic_size (int, optional): size of semantic size
        dropout (float, optional): dropout rate
    """

    def __init__(self, semantic_size=1000, dropout=0.15):
        super(EncoderTagger, self).__init__()
        self.semantic_size = semantic_size

        resnet = torchvision.models.resnet152(
            pretrained=True)  # pretrained ImageNet ResNet-152

        # Remove linear layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(
            *modules)

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(2048, semantic_size)

        self.sigmoid = nn.Sigmoid()

        self.fine_tune()

    def forward(self, images):
        r"""Forward propagation.

        Arguments
            images (torch.Tensor): images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        Returns 
            torch.Tensor: probabilites of tags (batch_size, 1000)
        """
        out = self.resnet(images)
        out = out.view(out.size(0), -1)   # (batch_size, 2048)
        out = self.dropout(out)    # (batch_size, 2048)
        out = self.linear(out)     # (batch_size, 1000)
        out = self.sigmoid(out)    # (batch_size, 1000)
        return out

    def fine_tune(self, fine_tune=True):
        r"""Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Arguments
            fine_tune (boolean): Allow fine tuning?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    r"""Attention Network.

    Arguments
        encoder_dim (int): feature size of encoded images
        decoder_dim (int): size of decoder's RNN
        attention_dim (int): size of the attention network
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        r"""Forward propagation.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
            decoder_hidden (torch.Tensor): previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        Returns
            torch.Tensor: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    r"""Caption Decoder with Attention Networks

    Arguments
        attention_dim (int): size of attention network
        embed_dim (int): embedding size
        decoder_dim (int): size of decoder's RNN
        vocab_size (int): size of vocabulary
        encoder_dim (int, optional): feature size of encoded images
        dropout (float, optional): dropout
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        r"""Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        r"""Loads embedding layer with pre-trained embeddings.

        Arguments
            embeddings (torch.Tensor): pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        r"""Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        Arguments
            fine_tune (boolean): Allow fine tuning?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        r"""Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        Returns 
            torch.Tensor: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        r"""Forward propagation.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoded_caption (torch.Tensor): encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths (torch.Tensor): caption lengths, a tensor of dimension (batch_size, 1)
        Returns
            (Tuple of torch.Tensor): scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :],
                           attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class SCNCell(nn.Module):
    r"""Custom Cell for Implementing Semantic Compositional Networks

    Arguments
        input_size (int): size of input
        hidden_size (int): size of embedding
        semantic_size (int): size of semantic
        factor_size (int): size of factor
        bias (boolean, optional): use bias?
    """

    def __init__(self, input_size, hidden_size, semantic_size, factor_size, bias=True):
        super(SCNCell, self).__init__()

        self.factor_size = factor_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.semantic_size = semantic_size

        self.weight_ia = nn.Parameter(
            torch.Tensor(input_size, 4 * factor_size))
        self.weight_ib = nn.Parameter(
            torch.Tensor(semantic_size, 4 * factor_size))
        self.weight_ic = nn.Parameter(
            torch.Tensor(hidden_size, 4 * factor_size))

        self.weight_ha = nn.Parameter(
            torch.Tensor(hidden_size, 4 * factor_size))
        self.weight_hb = nn.Parameter(
            torch.Tensor(semantic_size, 4 * factor_size))
        self.weight_hc = nn.Parameter(
            torch.Tensor(hidden_size, 4 * factor_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def forward(self, wemb_input, semantic_input, hx=None):
        self.check_forward_input(wemb_input)

        [ia_i, ia_f, ia_o, ia_c] = split_tensor2d(
            self.weight_ia, self.factor_size)
        [ib_i, ib_f, ib_o, ib_c] = split_tensor2d(
            self.weight_ib, self.factor_size)
        [ic_i, ic_f, ic_o, ic_c] = split_tensor2d(
            self.weight_ic, self.factor_size)
        [b_ii, b_if, b_io, b_ic] = split_tensor1d(
            self.bias_ih, self.hidden_size)

        tmp1_i = (wemb_input @ ia_i)
        tmp1_f = (wemb_input @ ia_f)
        tmp1_o = (wemb_input @ ia_o)
        tmp1_c = (wemb_input @ ia_c)

        tmp2_i = (semantic_input @ ib_i).unsqueeze(0)
        tmp2_f = (semantic_input @ ib_f).unsqueeze(0)
        tmp2_o = (semantic_input @ ib_o).unsqueeze(0)
        tmp2_c = (semantic_input @ ib_c).unsqueeze(0)

        state_below_i = ((tmp1_i * tmp2_i) @ ic_i.t()) + b_ii
        state_below_f = ((tmp1_f * tmp2_f) @ ic_f.t()) + b_if
        state_below_o = ((tmp1_o * tmp2_o) @ ic_o.t()) + b_io
        state_below_c = ((tmp1_c * tmp2_c) @ ic_c.t()) + b_ic

        x_i = state_below_i.squeeze(0)
        x_f = state_below_f.squeeze(0)
        x_o = state_below_o.squeeze(0)
        x_c = state_below_c.squeeze(0)

        if hx is None:
            hx = wemb_input.new_zeros(wemb_input.size(
                0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        self.check_forward_hidden(x_i, hx[0], '[0]')
        self.check_forward_hidden(x_i, hx[1], '[1]')

        self.check_forward_hidden(x_f, hx[0], '[0]')
        self.check_forward_hidden(x_f, hx[1], '[1]')

        self.check_forward_hidden(x_o, hx[0], '[0]')
        self.check_forward_hidden(x_o, hx[1], '[1]')

        self.check_forward_hidden(x_c, hx[0], '[0]')
        self.check_forward_hidden(x_c, hx[1], '[1]')

        return self.recurrent_step(x_i, x_f, x_o, x_c, semantic_input, hx)

    def recurrent_step(self, x_i, x_f, x_o, x_c, semantic_input, hx):
        h_, c_ = hx

        [ha_i, ha_f, ha_o, ha_c] = split_tensor2d(
            self.weight_ha, self.factor_size)
        [hb_i, hb_f, hb_o, hb_c] = split_tensor2d(
            self.weight_hb, self.factor_size)
        [hc_i, hc_f, hc_o, hc_c] = split_tensor2d(
            self.weight_hc, self.factor_size)
        [b_hi, b_hf, b_ho, b_hc] = split_tensor1d(
            self.bias_hh, self.hidden_size)

        preact_i = (h_ @ ha_i) * (semantic_input @ hb_i)
        preact_i = (preact_i @ hc_i.t()) + x_i + b_hi

        preact_f = (h_ @ ha_f) * (semantic_input @ hb_f)
        preact_f = (preact_f @ hc_f.t()) + x_f + b_hf

        preact_o = (h_ @ ha_o) * (semantic_input @ hb_o)
        preact_o = (preact_o @ hc_o.t()) + x_o + b_ho

        preact_c = (h_ @ ha_c) * (semantic_input @ hb_c)
        preact_c = (preact_c @ hc_c.t()) + x_c + b_hc

        i = torch.sigmoid(preact_i)
        f = torch.sigmoid(preact_f)
        o = torch.sigmoid(preact_o)
        c = torch.tanh(preact_c)

        c = f * c_ + i * c
        h = o * torch.tanh(c)

        return h, c

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class DecoderSCNWithAttention(nn.Module):
    r"""Caption Decoder with Semantic Compositional + Attention Networks

    Arguments
        attention_dim (int): size of attention network
        embed_dim (int): embedding size
        decoder_dim (int): size of decoder's RNN
        factored_dim (int): size of factorization
        semantic_dim (int): size of tag input
        vocab_size (int): size of vocabulary
        encoder_dim (int, optional): feature size of encoded images
        dropout (float, optional): dropout
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, factored_dim, semantic_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderSCNWithAttention, self).__init__()

        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.factored_dim = factored_dim
        self.semantic_dim = semantic_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = SCNCell(
            embed_dim + encoder_dim, decoder_dim, semantic_dim, factored_dim, bias=True)  # decoding SCNCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        r"""Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        r"""Loads embedding layer with pre-trained embeddings.

        Arguments
            embeddings (torch.Tensor): pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        r"""Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        Arguments
            fine_tune (boolean): Allow fine tuning?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        r"""Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        Returns 
            torch.Tensor: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, semantic_input, encoded_captions, caption_lengths):
        r"""Forward propagation.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
            semantic_input (torch.Tensor): encoded tags, a tensor of dimension (batch_size, semantic_size)
            encoded_caption (torch.Tensor): encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths (torch.Tensor): caption lengths, a tensor of dimension (batch_size, 1)
        Returns
            (Tuple of torch.Tensor): scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :],
                           attention_weighted_encoding], dim=1),
                semantic_input[:batch_size_t, :],
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
