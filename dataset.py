import os
import pickle
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.neighbors import KNeighborsTransformer

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

from tqdm import tqdm

from model.utils import NamedTensorDataset


class Images_Augmentation_Subset(torch.utils.data.dataset.Dataset):

    def __init__(self, orig_dataset: NamedTensorDataset,
                 pos_augments, num_pos,
                 return_index=True):
        self.orig_dataset = orig_dataset
        self.pos_augments = pos_augments
        self.num_pos = num_pos
        if self.num_pos == 0:
            self.num_pos = 1
        self.return_index = return_index
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, index):
        orig_item = self.orig_dataset[index]
        np_image = (orig_item['img'].detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        single_channel = False
        if np_image.shape[-1] == 1:
            single_channel = True
            np_image = np.concatenate([np_image] * 3, axis=-1)
        pil_image = Image.fromarray(np_image)
        pos_augments_indxs = np.random.choice(range(len(self.pos_augments)), self.num_pos, replace=False)
        pos_imgs = torch.cat(
            [self.to_tensor(self.pos_augments[i](pil_image)).unsqueeze(0) for i in pos_augments_indxs], dim=0
        )
        if single_channel:
            pos_imgs = torch.mean(pos_imgs, dim=1).unsqueeze(1)
        orig_item['pos_imgs'] = pos_imgs
        if self.return_index:
            orig_item['index'] = index
        return orig_item
