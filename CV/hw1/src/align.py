# coding: utf-8


import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    mu = min(10 / image.shape[0], 10. / image.shape[1])
    plt.figure(figsize=(mu * image.shape[1], mu * image.shape[0]))
    plt.imshow(image)
    plt.show()


def show_channels(red=None, green=None, blue=None):
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


def split_channels(image):
    length = image.shape[0] // 3
    return image[2 * length: 3 * length], image[length: 2 * length], image[:length]


def strip_frame(image):
    fr_len, fr_wi = int(0.05 * image.shape[0]), int(0.05 * image.shape[1])
    return image[fr_len: -fr_len, fr_wi:-fr_wi]


def preprocess_image(image):
    #     stripped_image = strip_frame(image)
    red, green, blue = split_channels(image.astype(np.float32))
    red, green, blue = strip_frame(red), strip_frame(green), strip_frame(blue)
    return red, green, blue



def align(image, g_coord, offset=15, sim_method=cv2.TM_CCORR_NORMED, best_res=max):
    prep_image = preprocess_image(image)

    red, green, blue = prep_image

    green_loc = (offset, offset)
    _, green_can = _locate_on_canvas(green, green_loc, offset)

    red_res = cv2.matchTemplate(green_can, red, sim_method)
    blue_res = cv2.matchTemplate(green_can, blue, sim_method)

    i = 1 if best_res == max else 2
    red_loc = cv2.minMaxLoc(red_res)[-i]
    blue_loc = cv2.minMaxLoc(blue_res)[-i]


    rc = np.array(red_loc[::-1])
    gc = np.array(green_loc[::-1])
    bc = np.array(blue_loc[::-1])
    img = show_channels(_locate_on_canvas(red, rc, offset)[1],
                  _locate_on_canvas(green, gc, offset)[1],
                  _locate_on_canvas(blue, bc, offset)[1])
    g_coord = np.array(g_coord) - np.array((green.shape[0], 0))
    # print(g_coord)
    # print(g_coord + gc - bc)
    # print(bc)
    fr_len = red.shape[0] / 0.95 * 0.05
    print(fr_len)
    ba = (g_coord + gc - bc)
    ra = (g_coord + gc - rc)
    print(gc, bc, rc)
    # ba[0] += fr_len
    print(g_coord, ba, ra)
    ra[0] += blue.shape[0] * 2
    # ba[0] +=  -4 * fr_len
    # ba[0] = 2 * blue.shape[0] - ba[0] + 10
    # ra[0] = red.shape[0] * 2 + ra[0] + 30
    return img, ba, ra

def convert_low_high(coords, im):
    return np.array([im.shape[0] - coords[0], coords[1]])

def _get_bbox_slice(ul, br):
    return slice(ul[0], br[0]), slice(ul[1], br[1])



def _locate_on_canvas(ch, ul, offset):
    canvas = np.full(
        (ch.shape[0] + 2 * offset, ch.shape[1] + 2 * offset), np.mean(ch), dtype=np.float32)
    br = (ul[0] + ch.shape[0], ul[1] + ch.shape[1])
    canvas[_get_bbox_slice(ul, br)] = ch
    return br, canvas


def _get_bbox(im, ul, br):
    return im[_get_bbox_slice(ul, br)]


