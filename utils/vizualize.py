import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage.transform

from PIL import Image


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    r"""Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    Arguments
        image_path: path to image that has been captioned
        seq: caption
        alphas: weights
        rev_word_map: reverse word mapping, i.e. ix2word
        smooth: smooth weights?
    """
    plt.figure(1, figsize=(24, 9))

    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black',
                 backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
