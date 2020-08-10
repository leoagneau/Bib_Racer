import numpy as np
import matplotlib.image as mpimg
import os
import glob
import time
import pickle
import bz2

def save_object(obj, filename):
    with open(filename,  'wb') as output:
        pickle.dump(obj, output)

def save_compressed_obj(obj, filename):
    with bz2.BZ2File(filename, 'wb') as output:
        pickle.dump(obj, output)

path = os.getcwd()
images = {key: [] for key in ['train', 'test']}
for dir in ['train', 'test']:
    print(dir)
    tmp_imgs = []
    files = glob.glob(os.path.join(path, dir, '*.png'))
    st = time.time()
    for file in files:
        img = mpimg.imread(file)
        tmp_imgs.append(img)
    et = time.time()
    print('finished reading', dir, 'in', '{:.5f}'.format(et - st))
    images[dir] = np.asarray(tmp_imgs)
    print('casting completed in', '{:.5f}'.format(time.time() - et), ', len:', len(images[dir]))

st = time.time()
save_compressed_obj(images['train'], 'train_images.pkl.bz2')
et = time.time()
print('finished saving training in:', '{:.5f}'.format(et - st))
save_compressed_obj(images['test'], 'test_images.pkl.bz2')
print('finished saving testing in:', '{:.5f}'.format(time.time() - et))