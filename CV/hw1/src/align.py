# coding: utf-8


import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def split_channels(image):
    length = image.shape[0] // 3
    return image[2 * length: 3 * length], image[length: 2 * length], image[:length]


def calc_frame(image, perc=0.05):
    fr_len, fr_wi = int(perc * image.shape[0]), int(perc * image.shape[1])
    return fr_len, fr_wi


def strip_frame(image, fr_len, fr_wi):
    return image[fr_len: -fr_len, fr_wi:-fr_wi]


def preprocess_image(image):
    red, green, blue = split_channels(image.astype(np.float32))
    fr_len, fr_wi = calc_frame(red, 0.05)
    red, green, blue = strip_frame(red, fr_len, fr_wi), strip_frame(green, fr_len, fr_wi), strip_frame(blue, fr_len,
                                                                                                       fr_wi)
    return red, green, blue, fr_len, fr_wi


# def cut_channels(red, green, blue, rc, gc, bc):
#     rca, gca, bca = aggregate_ul(rc, gc, bc)
#     print(rca, gca, bca)
#     red_a = red[rca[0]:, rca[1]:]
#     green_a = green[gca[0]:, gca[1]:]
#     blue_a = blue[bca[0]:, bca[1]:]
#
#     shx, shy = (min((ch.shape[0] for ch in (red_a, green_a, blue_a))),
#                     min((ch.shape[1] for ch in (red_a, green_a, blue_a))))
#
#     return red_a[:shx, :shy], green_a[:shx, :shy], blue_a[:shx, :shy]

def cut_channels(red, green, blue, rc, gc, bc):
    rca, gca, bca = aggregate_ul(rc, gc, bc)
    print(rca, gca, bca)
    shx, shy = red.shape

    shx, shy = (min((shx - ca[0] for ca in (rca, gca, bca))),
                    min((shy - ca[1] for ca in (rca, gca, bca))))

    return (red[rca[0]:rca[0] + shx, rca[1]:rca[1] + shy],
            green[gca[0]:gca[0] + shx, gca[1]:gca[1] + shy],
            blue[bca[0]:bca[0] + shx, bca[1]:bca[1] + shy])


def aggregate_ul(rc, gc, bc):
    gca = max((rc[0], gc[0], bc[0])), max((rc[1], gc[1], bc[1]))
    rca = max((0, gca[0] - rc[0])), max((0, gca[1] - rc[1]))
    bca = max((0, gca[0] - bc[0])), max((0, gca[1] - bc[1]))
    return rca, gca, bca




def align(image, g_coord, offset=15, sim_method=cv2.TM_CCORR_NORMED, best_res=max):
    if image.shape[1] / 1000 > 1:
        offset = 80
    red, green, blue, fr_len, fr_wi = preprocess_image(image)

    def get_bbox(im):
        return im[offset:-offset, offset:-offset]

    red_res = cv2.matchTemplate(green, get_bbox(red), sim_method)
    blue_res = cv2.matchTemplate(green, get_bbox(blue), sim_method)

    i = 1 if best_res == max else 2
    red_loc = cv2.minMaxLoc(red_res)[-i]
    blue_loc = cv2.minMaxLoc(blue_res)[-i]

    print(red_loc, blue_loc)

    rc = np.array(red_loc[::-1]) - offset
    bc = np.array(blue_loc[::-1]) - offset
    gc = (0, 0)

    img = stack_channels(*cut_channels(red, green, blue, rc, gc, bc))

    # get relative green coordinates
    g_coord = np.array(g_coord) - np.array((red.shape[0] + 3 * fr_len, 0))

    # get relative blue & red coordinates
    b_coord = (g_coord + gc - bc)
    r_coord = (g_coord + gc - rc)

    # convert to absolute
    r_coord[0] += blue.shape[0] * 2 + 5 * fr_len
    b_coord[0] += fr_len

    return img, b_coord, r_coord


def _get_bbox_slice(ul, br):
    return slice(ul[0], br[0]), slice(ul[1], br[1])


def _locate_on_canvas(ch, ul, offset):
    canvas = np.full(
        (ch.shape[0] + 2 * offset, ch.shape[1] + 2 * offset), np.mean(ch), dtype=np.float32)
    br = (ul[0] + ch.shape[0], ul[1] + ch.shape[1])
    canvas[_get_bbox_slice(ul, br)] = ch
    return br, canvas

