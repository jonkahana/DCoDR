
from os.path import join as join

import cv2
from tqdm import tqdm

from PIL import Image
import glob
from copy import deepcopy

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import disentanglement_lib as d_lib
from disentanglement_lib.data.ground_truth.cars3d import CARS3D_PATH


save_folder = 'cache/preprocess'
os.makedirs(save_folder, exist_ok=True)

np.random.seed(2)

def read_images(imgs):
    classes = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
    contents = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)

    for elevation in range(4):
        for azimuth in range(24):
            for object_id in range(183):
                img_idx = elevation * 24 * 183 + azimuth * 183 + object_id

                classes[img_idx] = object_id
                contents[img_idx] = elevation * 24 + azimuth

    return imgs, classes, contents

if __name__ == '__main__':

    d_lib.data.ground_truth.cars3d.CARS3D_PATH = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "cars")
    d_lib_cars = d_lib.data.ground_truth.cars3d.Cars3D()
    imgs, classes, contents = read_images(d_lib_cars.images)
    imgs = (imgs * 255.).astype(np.uint8)
    n_classes = len(np.unique(classes))

    # seed of numpy is CONSTANT to ensure the same stratified(!) split
    train_indxs, test_indxs = train_test_split(np.arange(len(classes)), stratify=classes, test_size=0.1)
    train_indxs = train_indxs.astype(int)
    test_indxs = test_indxs.astype(int)

    # region Save Train Dataset

    train_imgs = imgs[train_indxs]
    train_contents = contents[train_indxs]
    train_classes = classes[train_indxs]
    np.savez(join(save_folder, f'cars3d_train.npz'),
             imgs=train_imgs,
             contents=train_contents,
             classes=train_classes,
             n_classes=n_classes
             )

    # endregion Save Train Dataset

    # region Save Test Dataset

    test_imgs = imgs[test_indxs]
    test_contents = contents[test_indxs]
    test_classes = classes[test_indxs]
    np.savez(join(save_folder, f'cars3d_test.npz'),
             imgs=test_imgs,
             contents=test_contents,
             classes=test_classes,
             n_classes=n_classes
             )

    # endregion Save Test Dataset
