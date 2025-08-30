# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import numpy as np
from torchvision import transforms
from PIL import Image
import subprocess
from torch.utils.data import Dataset


import datasets
import datasets.base

class Pendulum(Dataset):

    lat_names = ('PendAngle', 'LightPos', 'ShadowLength', 'ShadowPos')
    lat_sizes = np.array([1, 3, 6, 40, 32, 32])

    img_size = (3, 64, 64) # orinal size: (3, 96, 96), required to be (, 64, 64) by model
    background_color = datasets.COLOUR_WHITE

    lat_values = {
        'ShadowLength':
        np.array([
            0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032,
            0.19354839, 0.22580645, 0.25806452, 0.29032258, 0.32258065,
            0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
            0.51612903, 0.5483871, 0.58064516, 0.61290323, 0.64516129,
            0.67741935, 0.70967742, 0.74193548, 0.77419355, 0.80645161,
            0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.
        ]),
        'LightPos':
        np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
        'ShadowPos':
        np.array([
            0., 0.16110732, 0.32221463, 0.48332195, 0.64442926, 0.80553658,
            0.96664389, 1.12775121, 1.28885852, 1.44996584, 1.61107316,
            1.77218047, 1.93328779, 2.0943951, 2.25550242, 2.41660973,
            2.57771705, 2.73882436, 2.89993168, 3.061039, 3.22214631,
            3.38325363, 3.54436094, 3.70546826, 3.86657557, 4.02768289,
            4.1887902, 4.34989752, 4.51100484, 4.67211215, 4.83321947,
            4.99432678, 5.1554341, 5.31654141, 5.47764873, 5.63875604,
            5.79986336, 5.96097068, 6.12207799, 6.28318531
        ]),
        'PendAngle':
        np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
    }

    def __init__(self, root='data/causal_data/data/pendulum', split="train", **kwargs):

        super().__init__()
        
        root = root + "/" + split
        imgs = os.listdir(root)

        self.split = split
        
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        
        label = np.asarray(self.imglabel)
        self.label_avg = np.mean(label, axis=0)
        self.label_std = np.std(label, axis=0) 
        
        
        #print(self.imglabel)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((64,64)),
                transforms.ToTensor()
             ]
            )


    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.array(pil_img)
        array1 = np.array(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(3,64,64)
            data = torch.from_numpy(pil_img)
        
        # print(label)

        return data, label.float()

    def __len__(self):
        return len(self.imgs)
