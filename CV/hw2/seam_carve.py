import matplotlib.pyplot as plt
import numpy as np
import cv2


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


def _seam_carve(image, mode='', mask=None):
    #     Y = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:, :, 0]
    Y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    y_ = np.vstack((Y[[1]] - Y[[0]], Y[2:] - Y[:-2], Y[[-1]] - Y[[-2]])).astype(np.float64)
    x_ = np.hstack((Y[:, [1]] - Y[:, [0]], Y[:, 2:] - Y[:, :-2], Y[:, [-1]] - Y[:, [-2]])).astype(np.float64)
    #     y_ = cv2.Sobel(Y, cv2.CV_64F, 0, 1)
    #     x_ = cv2.Sobel(Y, cv2.CV_64F, 1, 0)

    if mask is None:
        mask = np.zeros((image.shape[0], image.shape[1]))
    else:
        mask  = ((mask/255) - 0.5) * 2
    # elif mask.shape[2] == 4:
    #     mask = mask[:, :, 3]
    # else:
    #     mask = np.zeros((image.shape[0], image.shape[1]))
    # print(mask.shape, image.shape)
    # print((mask[:, :, 3] * image.shape[0] * image.shape[1]).shape)
    gradient = np.sqrt(x_ ** 2 + y_ ** 2) + mask * image.shape[0] * image.shape[1]

    def argmin_on_slice(arr, i_from, i_to):
        if i_from < 0:
            return np.argmin(arr[0:i_to])
        elif i_to > len(arr):
            return np.argmin(arr[i_from:]) + i_from
        else:
            return np.argmin(arr[i_from:i_to]) + i_from

    def min_filter(image):
        result = np.copy(image)
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1]):
                result[i + 1, j] += min(result[i, max(j - 1, 0):min(image.shape[1], j + 2)])
        return result

    matrix = min_filter(gradient)

    def resolve_seam(matrix):
        seam = []
        seam.append(np.argmin(matrix[-1]))
        for i in range(2, matrix.shape[0] + 1):
            j_prev = seam[-1]
            j = argmin_on_slice(matrix[-i], j_prev - 1, j_prev + 2)
            seam.append(j)
        mask = np.full(matrix.shape, False)
        for i in zip(range(matrix.shape[0]), seam[::-1]):
            mask[i] = True
        return mask

    seam = resolve_seam(matrix)
 
    return image[~seam].reshape((image.shape[0], -1, 3)), mask, seam


def seam_carve(image, mode='', mask=None):
    if mode.startswith('horizontal'):
        return _seam_carve(image, mode, mask)
    elif mode.startswith('vertical'):
        if mask is not None:
            mask = mask.transpose((1, 0))
        im, mask, seam = _seam_carve(image.transpose((1, 0, 2)), mode, mask)
        if mask is not None:
            mask = mask.transpose((1, 0))
        return im.transpose((1, 0, 2)), mask, seam.transpose((1, 0))
    return _seam_carve(image, mode, mask)