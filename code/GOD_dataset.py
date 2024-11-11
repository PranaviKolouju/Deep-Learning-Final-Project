import numpy as np
import os
import json
import csv
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy import interpolate
import torchvision.transforms as transforms

# Helper functions
def identity(x):
    return x

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size - x.shape[1] % patch_size)), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def img_norm(img):
    img = torch.tensor(img)
    return (img / 255.0) * 2.0 - 1.0

def get_img_label(class_index, img_filename, naive_label_set=None):
    img_label = []
    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(class_index.values(), class_index.keys())):
            if name in w[:-1]:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

# Kamitani Dataset
def create_Kamitani_dataset(path, roi='VC', patch_size=16, fmri_transform=identity,
                            image_transform=identity, subjects=['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5']):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz')))
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]

    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    train_img = []
    train_fmri = []
    train_img_label_all = []
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))
        roi_mask = npz[roi]
        tr = normalize(pad_to_patch_size(npz['arr_0'][..., roi_mask], patch_size))
        train_fmri.append(tr)
        train_img.append(img_npz['train_images'][npz['arr_3']])
        train_img_label_all += [train_img_label[i] for i in npz['arr_3']]

    train_fmri = np.concatenate(train_fmri, axis=0)
    train_img = np.concatenate(train_img, axis=0)
    return Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform)

class Kamitani_dataset(Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity):
        super(Kamitani_dataset, self).__init__()
        self.fmri = np.expand_dims(fmri, axis=1) # (samples, 1, voxels)
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri_transform(self.fmri[index])
        img = self.image_transform(self.image[index] / 255.0)
        return {'fmri': fmri, 'image': img}