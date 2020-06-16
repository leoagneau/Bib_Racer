############################
### Prepare SVHN dataset ###
############################
import os
import h5py
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
from operator import add


### Get number of total instances in digitStruct.mat
def get_instances_num(f):
    return f['digitStruct/names'].shape[0]


### Get filename from index
# https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
# https://stackoverflow.com/a/56388672/3243870
def get_img_name(f, idx=0):
    names = f['digitStruct/names']
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return img_name


### Get bounding box from index
# elements in bbox struct: height, width, top, left, label
bbox_prop = ['height', 'left', 'top', 'width', 'label']
def get_img_boxes(f, idx=0):
    """
    Get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    meta = {key: [] for key in bbox_prop}
    bboxs = f['digitStruct/bbox']

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta


def merge_bbox(f, idx=0):
    """
    Return a bounding box that includes all the individual bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary contains the properties of the bounding box
    """
    meta = get_img_boxes(f, idx)
    # print(meta)
    left = min(meta['left'])
    top = min(meta['top'])
    width = max(map(add, meta['left'], meta['width'])) - left
    height = max(map(add, meta['top'], meta['height'])) - top
    labels = [x if x != 10 else 0 for x in meta['label']]
    bbox = {'left': left, 'top': top, 'width': width, 'height': height, 'labels': labels}
    return bbox


def save_object(obj, filename):
    with open(filename,  'wb') as output:
        pickle.dump(obj, output, -1)


def process_digitStruct(f):
    """
    Merge all the individual bounding boxes of each image in the dataset
    :param f: h5py.File
    :return: a dictionary where the key is file name and value is the properties
    of the corresponding bounding box
    """
    data = {}
    num_items = get_instances_num(f)
    for idx in range(num_items):
        if idx % 100 == 0:
            print(idx)
        bbox = merge_bbox(f, idx)
        fname = get_img_name(f, idx)
        data[fname] = bbox
        # print(bbox, labels)
    print(len(data))
    return data


def main():
    path = os.getcwd()
    train_path = os.path.join(path, 'train')
    train_metafile = os.path.join(train_path, 'digitStruct.mat')
    tr_f = h5py.File(train_metafile, 'r')
    test_path = os.path.join(path, 'test')
    test_metafile = os.path.join(test_path, 'digitStruct.mat')
    te_f = h5py.File(test_metafile, 'r')
    PREP = False

    if PREP:
        ### To read digitStruct.mat and image files separately
        train_data = process_digitStruct(tr_f)
        save_object(train_data, 'train.pkl')
        print("Finished processing training data")

        test_data = process_digitStruct(te_f)
        save_object(test_data, 'test.pkl')
        print("Finished processing testing data")

    else:  # Show some examples
        # Open processed file
        TRAIN = True
        if TRAIN:
            datafile = os.path.join(train_path, 'train.pkl')
            imgpath = train_path
            f = tr_f
        else:
            datafile = os.path.join(test_path, 'test.pkl')
            imgpath = test_path
            f = te_f

        DATA_EXIST = False
        with open(datafile, 'rb') as input:
            data = pickle.load(input)
            DATA_EXIST = True

        if DATA_EXIST:
            for _ in range(10):
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

                # Draw individual bounding boxes
                meta = get_img_boxes(f, idx-1)
                print(meta)
                for i in range(len(meta['top'])):
                    ax.add_patch(patches.Rectangle((meta['left'][i], meta['top'][i]),
                                                   meta['width'][i], meta['height'][i],
                                                   linewidth=1, edgecolor='b', facecolor='none'))

                # Draw the outer bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.imshow(img)
                plt.title(data[fn]['labels'])
                plt.show()


if __name__ == "__main__":
    main()
