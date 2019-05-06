import torch
from torch import nn
import torchvision


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
