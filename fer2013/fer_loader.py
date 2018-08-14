# -*- coding: utf-8 -*-
"""Data loader for the Fer 2013 emotion dataset described
in the paper:

Goodfellow, Ian J., et al. "Challenges in representation learning:
A report on three machine learning contests." International Conference on
Neural Information Processing. Springer, Berlin, Heidelberg, 2013.
https://arxiv.org/abs/1307.0414
"""

import os
import csv
import tqdm
import torch
import pickle
import numpy as np
import PIL.Image

from os.path import join as pjoin

class Fer2013Dataset(torch.utils.data.Dataset):
    """Dataset class helper for the Fer2013 dataset. Converts the csv
    files used to distribute the dataset into a pikle format

    Args:
        data_dir (str): Directory where the original csv files distributed
            with the dataset are found.
        mode (str): The subset of the dataset to use
        transform (torch.transforms): a transformaton that can be applied
            to images on loading
        include_train (bool) [False]: whether to include the training set
            in the loader (it's not required for benchmarking purposes).
    """
    def __init__(self, data_dir, mode='val', transform=None,
                 include_train=False):
        self.data_dir = data_dir
        self.mode = mode
        self.include_train = include_train
        self._transform = transform
        self.pkl_path = pjoin(data_dir, 'pytorch', 'data.pkl')

        if not os.path.isfile(self.pkl_path):
            self.prepare_data()

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        """Retreive the sample at the given index.

        Args:
            index (int): the index of the sample to be retrieved

        Returns:
            (torch.Tensor): the image
            (int): the label
        """
        im_data = self.data['images'][self.mode][index].astype('uint8')
        image = PIL.Image.fromarray(im_data)
        label = self.data['labels'][self.mode][index]
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def prepare_data(self):
        """Transform raw data from csv format into a dict.

        Args:
            phase, str: 'train'/'val'/'test'.
            size, int. Size of the dataset.
        """
        print('preparing data...')
        with open(pjoin(self.data_dir, 'fer2013.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) # skip header
            rows = [row for row in reader]

        train_ims, val_ims, test_ims = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for row in tqdm.tqdm(rows):
            subset = row[2]
            raw_im = np.array([int(x) for x in row[1].split(' ')])
            im = np.repeat(raw_im.reshape(48,48)[:,:,np.newaxis], 3, axis=2)
            if subset == 'Training':
                train_labels.append(int(row[0]))
                train_ims.append(im)
            elif subset == 'PublicTest':
                val_labels.append(int(row[0]))
                val_ims.append(im)
            elif subset == 'PrivateTest':
                test_labels.append(int(row[0]))
                test_ims.append(im)
            else:
                raise ValueError('unrecognised subset: {}'.format(subset))

        data = {'labels': {}, 'images': {}}
        data['labels']['val'] = np.array(val_labels)
        data['labels']['test'] = np.array(test_labels)

        data['images']['val'] = np.array(val_ims)
        data['images']['test'] = np.array(test_ims)

        if self.include_train:
            data['labels']['train'] = np.array(train_labels)
            data['images']['train'] = np.array(train_ims)

        for key in 'images', 'labels':
            assert len(data[key]['val']) == 3589, 'unexpected length'
            assert len(data[key]['test']) == 3589, 'unexpected length'
            if self.include_train:
                assert len(data[key]['train']) == 28709, 'unexpected length'

        if not os.path.exists(os.path.dirname(self.pkl_path)):
                os.makedirs(os.path.dirname(self.pkl_path))

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self):
        """Return the total number of images in the datset.

        Return:
            (int) the number of images.
        """
        return self.data['labels'][self.mode].size

