############################
### Prepare SVHN dataset ###
############################

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms as T
import time
import copy

BS=16
path = os.getcwd()
### To read digitStruct.mat and image files separately
# img_path = os.path.join(path, 'SVHN', 'images')
# digi_file = os.path.join(img_path, 'digitStruct.mat')
# f = h5py.File(digi_file, 'r')
# names = f['digitStruct/name']
# bboxs = f['digitStruct/bbox']

### Get filename from index
# https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
# https://stackoverflow.com/a/56388672/3243870
def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)

### Get bounding box from index
# elements in bbox struct: height, width, top, left, label
bbox_prop = ['height', 'left', 'top', 'width', 'label']
def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    meta = { key : [] for key in bbox_prop}

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta


import matplotlib.pyplot as plt
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def save_check_pt(epoch, best_model_wts, optimizer, best_acc, PATH):
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, PATH)


def train_model(dataloaders, dataset_sizes, device, model, criterion, optimizer, scheduler, num_epochs=25, test=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            iter = 0
            itertotal = dataset_sizes[phase] //BS + 1
            epoch_start_T = time.time()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                iter += 1
                if ((iter - 1) % 100 == 0):
                    epoch_elapsed = time.time() - epoch_start_T
                    print('{}/{}, time elapsed: {:0f}m {:0f}s'.format(iter, itertotal, epoch_elapsed // 60,
                                                                      epoch_elapsed % 60))
                if (test and iter == 3):
                    print(iter)
                    break
            if (phase == 'train'):
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if (phase == 'test' and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if (((epoch + 1) % 10) == 0):
            save_check_pt(epoch, best_model_wts, optimizer, best_acc, os.path.join(path, 'densenet_weights_{}.pth'.format(epoch+1)))

        print()

    time_elapsed = time.time() - since
    print('Training completes in {:0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return(model)


def main():
    # To deal with num_workers>1 in DataLoader, which is needed only in Windows environment
    # https://github.com/pytorch/pytorch/issues/11139
    # https://stackoverflow.com/questions/33038209/python-multiprocessing-fails-for-example-code
    import multiprocessing
    multiprocessing.freeze_support()

    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': T.Compose([
            T.RandomResizedCrop(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    SVHN_path = os.path.join(path, 'mat')
    SVHN = {x:datasets.SVHN(SVHN_path, split=x, transform=data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(SVHN[x], batch_size=BS, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(SVHN[x]) for x in ['train', 'test']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print('Device:', device)

    # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[x for x in classes])

    ########################
    ### Prepare DenseNet ###
    ########################

    densenet = models.densenet161(pretrained=True)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, 10)
    densenet = densenet.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    densenet = train_model(dataloaders, dataset_sizes, device, densenet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1, test=True)
    torch.save(densenet.state_dict(), os.path.join(path, 'densenet_weights_final.pth'))

if __name__ == "__main__":
    main()
