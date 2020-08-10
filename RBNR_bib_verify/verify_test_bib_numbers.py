import argparse
import fnmatch
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

def press(event):
    if event.key == 'n':
        plt.close()
    if event.key == 'x':
        sys.exit()

def verify_bib_numbers(path):
    bib_number_file = 'bib_numbers.txt'
    with open(os.path.join(path, bib_number_file)) as f:
        bib_numbers = f.read().split('\n')
    dpi = mpl.rcParams['figure.dpi']
    i = 0
    dd = os.scandir(path)

    # files = fnmatch.filter(os.scandir(path), '*.jpg')
    files = [n for n in dd if fnmatch.fnmatch(n, '*.jpg')]
    filenum = len(files)
    print('Number of images: {:d}'.format(filenum))
    for entry in files:
        print(entry.path)
        img = mpimg.imread(entry.path)
        height, width, depth = img.shape
        margin = 50
        figsize = (width + margin*2) / float(dpi), (height + margin*2) / float(dpi)
        left = margin / dpi / figsize[0]  # axes ratio
        bottom = margin / dpi / figsize[1]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom)
        fig.canvas.mpl_connect('key_press_event', press)

        plt.xticks([])
        plt.yticks([])
        plt.title("{:d}: {:s}".format(i+1, bib_numbers[i]))
        plt.xlabel("w:{:d}, h:{:d}".format(width, height))
        plt.imshow(img)
        plt.show()
        i += 1

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

verify_bib_numbers(args.path)