import torch
from torch import nn
import torchvision


class ImageTagger(nn.Module):
    """
    Image Tagger
    """

    def __init__(self, bottleneck_size, semantic_size, dropout=0.3):
        super(ImageTagger, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.semantic_size = semantic_size
        self.dropout = dropout

        self.fc = nn.Linear(bottleneck_size, semantic_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, bottleneck):
        predictions = self.fc(bottleneck)
        predictions = self.dropout(predictions)
        predictions = self.sigmoid(predictions)
        return predictions
