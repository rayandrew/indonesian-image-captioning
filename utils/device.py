import torch


def get_device():
    r"""Get current Pytorch device

    Return:
      Pytorch Device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
