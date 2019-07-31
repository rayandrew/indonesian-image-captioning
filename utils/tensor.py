def split_tensor1d(tensor, split):
    r"""Split 1D of tensor into N split

    Arguments
        tensor (Pytorch.Tensor) : tensor to split
        split  (int) : number of split
    Return
        array of splitted tensor with N elements of array
    """
    return [
        tensor[:split],
        tensor[split: split * 2],
        tensor[split * 2: split * 3],
        tensor[split * 3:],
    ]


def split_tensor2d(tensor, split, front=False):
    r"""Split 2D of tensor into N split of 2D tensor

    Arguments
        tensor (Pytorch.Tensor) : tensor to split
        split  (int) : number of split
        front  (bool) : split axis 0 if True else axis 1
    Return
        array of splitted 2D tensor with N elements of array
    """

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
