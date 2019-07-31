import torch
from torch import nn
import torchvision


class EncoderCaption(nn.Module):
    r"""Image Encoder extends ResNet152 model.

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


class Encoder(EncoderCaption):
    # fallback for old model
    pass
