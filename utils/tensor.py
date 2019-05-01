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
