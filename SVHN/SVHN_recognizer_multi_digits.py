import os
import pickle
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path = os.getcwd()
train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

# Load bounding box data
TRAIN = True
if TRAIN:
    datafile = os.path.join(train_path, 'train.pkl')
    imgpath = train_path
else:
    datafile = os.path.join(test_path, 'test.pkl')
    imgpath = test_path

with open(datafile, 'rb') as input:
    data = pickle.load(input)



SHOW_SAMPLES = True
if SHOW_SAMPLES:
    for _ in range(4):
        idx = random.randint(1, len(data))
        fn = '{}{}'.format(idx, '.png')
        # Display image
        x = data[fn]['left']
        y = data[fn]['top']
        w = data[fn]['width']
        h = data[fn]['height']
        img_f = os.path.join(imgpath, fn)
        img = mpimg.imread(img_f)
        fig, ax = plt.subplots(1)
        # Draw the outer bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.imshow(img)
        plt.title(data[fn]['labels'])
        plt.show()

# len_train = len(train_images)
# rp = np.random.permutation(len_train)
# train_images = train_images[rp]
