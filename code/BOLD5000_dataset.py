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

# BOLD5000 Dataset
def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.txt'):
                    sti_name += list(np.loadtxt(os.path.join(folder_path, file), dtype=str))
    return [name.replace('rep_', '', 1) if name.startswith('rep_') else name for name in sti_name]

def create_BOLD5000_dataset(path, patch_size=16, fmri_transform=identity, image_transform=identity, subjects=['CSI1', 'CSI2', 'CSI3', 'CSI4']):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'), allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_train_major, fmri_test_major, img_train_major, img_test_major = [], [], [], []
    for sub in subjects:
        fmri_data_sub = []
        for roi in roi_list:
            npy = f'{sub}_{roi}.npy'
            fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = normalize(pad_to_patch_size(np.concatenate(fmri_data_sub, axis=-1), patch_size))
        
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        test_idx = [img_files.index(img) for img in repeated_imgs_list if img in img_files]
        
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx] for idx in test_idx])
        
        train_idx = [i for i in range(len(img_files)) if i not in test_idx]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
        
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)
    
    return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform), 
            BOLD5000_dataset(fmri_test_major, img_test_major, fmri_transform, image_transform))

class BOLD5000_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity):
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
