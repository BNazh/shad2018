import matplotlib.pyplot as plt
import numpy as np


def seam_carve(image, mode, mask=None):
    pass


def show_image(image):
    mu = min(10 / image.shape[0], 10. / image.shape[1])
    plt.figure(figsize=(mu * image.shape[1], mu * image.shape[0]))
    plt.imshow(image)
    plt.show()


def stack_channels(red=None, green=None, blue=None):
    def _fill_zeros(red, green, blue):
        if red is None:
            if green is None:
                red = np.zeros_like(blue)
            else:
                red = np.zeros_like(green)
        return red

    red = _fill_zeros(red, green, blue)
    green = _fill_zeros(green, blue, red)
    blue = _fill_zeros(blue, red, green)
    im = np.dstack((red, green, blue))
    # show_image(im)
    return im


def show_channels(red=None, green=None, blue=None):
    im = stack_channels(red, green, blue)
    show_image(im)
    return im
