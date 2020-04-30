#! python

import argparse
import sys

import numpy as np
from PIL import Image

scl = 1.0

symbols = [
    (0.01, '@'),
    (0.005, '#'),
    (0.002, '-'),
    (0.001, '.'),
    (0.0, ' ')
]


def get2d(im, x, y):
    x0 = np.floor(x - 0.5).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y - 0.5).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    wx = x - x0 - 0.5
    wy = y - y0 - 0.5

    wa = (1.0 - wx) * (1.0 - wy)
    wb = (1.0 - wx) * wy
    wc = wx * (1.0 - wy)
    wd = wx * wy

    wa = wa.reshape(wa.shape + (1,))
    wb = wb.reshape(wb.shape + (1,))
    wc = wc.reshape(wc.shape + (1,))
    wd = wd.reshape(wd.shape + (1,))

    return wa * im[y0, x0] + wb * im[y1, x0] + wc * im[y0, x1] + wd * im[y1, x1]


def main():
    parser = argparse.ArgumentParser(description='converts image to text')
    parser.add_argument('scl', type=float, help='text scl')
    parser.add_argument('filename', help='filename of the image to be converted')

    args = parser.parse_args()
    filename = args.filename
    image = np.asarray(Image.open(filename), dtype='f8') / 255.0
    w, h = (np.array(image.shape[:2]) * args.scl).astype(int)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = (xx + 0.5) / w * image.shape[1]
    yy = (yy + 0.5) / h * image.shape[0]
    small = get2d(image, xx, yy)
    for y in range(h):
        for x in range(w):
            diff = 0.0
            count = 0
            if x != 0:
                d = small[y][x] - small[y][x - 1]
                diff += np.sum(d * d)
                count += 1
            if x != w - 1:
                d = small[y][x] - small[y][x + 1]
                diff += np.sum(d * d)
                count += 1
            if y != 0:
                d = small[y][x] - small[y - 1][x]
                diff += np.sum(d * d)
                count += 1
            if y != h - 1:
                d = small[y][x] - small[y + 1][x]
                diff += np.sum(d * d)
                count += 1
            diff /= count
            for value, c in symbols:
                if diff >= value * scl:
                    sys.stdout.write(c)
                    break
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()
