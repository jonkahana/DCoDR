import numpy as np
import pandas as pd

from tqdm import tqdm
import os
from os.path import join
import glob
import pickle

import matplotlib.pyplot as plt
import seaborn
import plotly.express as px
import cv2
from PIL import Image
from copy import deepcopy
from scipy.io import loadmat

import torch
from torch import cdist

from sklearn.model_selection import train_test_split


data_folder = 'raw_data'
save_folder = 'cache/preprocess'
os.makedirs(save_folder, exist_ok=True)


# Load Zappos-50K data

mat_path = join(data_folder, 'zappos_50k', 'ut-zap50k-data/image-path.mat')
if not os.path.exists(mat_path):
    raise ValueError('Run downlod_e2s_zappos.sh script before running this one!')
a = loadmat(mat_path)
if not os.path.exists(join(data_folder, 'zappos_50k', 'ut-zap50k-images')):
    raise ValueError('Run downlod_e2s_zappos.sh script before running this one!')
zap_relative_paths = [join(data_folder, 'zappos_50k', 'ut-zap50k-images', x[0]) for x in a['imagepath'][:, 0]]


def zap_load_img(path):
    img = np.array(Image.open(path))
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    return img


def zap_load_img_orig(path):
    img = np.array(Image.open(path))
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    return img


def enumerate_cols(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    all_maps = {}
    for col in columns:
        mapping = {x: i for i, x in enumerate(sorted(np.unique(df[col])))}
        df[col] = df[col].apply(lambda x: mapping[x])
        all_maps[col] = mapping
    return df, all_maps


zap_imgs = np.concatenate([zap_load_img(x)[np.newaxis, :, :, :] for x in tqdm(zap_relative_paths)])
zap_orig_imgs = np.concatenate([zap_load_img_orig(x)[np.newaxis, :, :, :] for x in tqdm(zap_relative_paths)])

zap_labels = pd.read_csv(join(data_folder, 'zappos_50k', 'ut-zap50k-data/meta-data.csv'))
final_labels = deepcopy(zap_labels[['Category', 'Gender', 'Closure', 'Material', 'ToeStyle']])
for col in final_labels.columns:
    final_labels[col] = final_labels[col].fillna('Other').apply(lambda x: x.split(';')[0])
    label_counts = final_labels[col].value_counts()
    final_labels.loc[(label_counts[final_labels[col]] <= 250).values, col] = 'Other'
final_labels = final_labels.rename({col: col.lower() for col in final_labels.columns}, axis=1)
final_labels = final_labels.rename({'category': 'shoe_type', 'toestyle': 'toe_style'}, axis=1)
final_labels__strings = deepcopy(final_labels)
final_labels, _ = enumerate_cols(final_labels)


# Load Edges2Shoes data

if not os.path.exists(os.path.join(data_folder, 'pix2pix', 'edges2shoes')):
    raise ValueError('Run downlod_e2s_zappos.sh script before running this one!')
e2s_image_files = glob.glob(os.path.join(data_folder, 'pix2pix', 'edges2shoes', '*', '*.jpg'))
e2s_image_files = np.asarray(e2s_image_files)


def e2s_num_sort(s):
    return int(s.split('/')[-1].split('_')[0])


e2s_image_files = sorted(e2s_image_files, key=e2s_num_sort)
len(e2s_image_files)


def e2s_extract_images(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    obj_img = deepcopy(img[:, 256:512])
    mask_img = deepcopy(img[:, 0:256])

    reduced_obj_img = cv2.resize(obj_img, (32, 32), interpolation=cv2.INTER_CUBIC)
    reduced_mask_img = cv2.resize(mask_img, (32, 32), interpolation=cv2.INTER_CUBIC)
    return obj_img, mask_img, reduced_mask_img, reduced_obj_img


def e2s_get_images(paths):
    imgs = []
    red_obj_imgs = []
    for fpath in tqdm(paths):
        obj_img, mask_img, red_mask_img, red_obj_img = e2s_extract_images(fpath)
        obj_img = np.array(obj_img)
        mask_img = np.array(mask_img)
        imgs.append(obj_img)
        imgs.append(mask_img)
        red_obj_imgs.append(red_obj_img)
    return imgs, red_obj_imgs


e2s_imgs, e2s_red_imgs = e2s_get_images(e2s_image_files)
classes = np.arange(len(e2s_imgs)) % 2
n_classes = len(np.unique(classes))

# Extract Features From Images

def extract_img_features(img):
    return img.flatten()


zap_features = np.array([extract_img_features(x) for x in tqdm(zap_imgs)])
e2s_features = np.array([extract_img_features(x) for x in tqdm(e2s_red_imgs)])
zap_features_t = torch.tensor(zap_features).double().to('cuda')
e2s_features_t = torch.tensor(e2s_features).double().to('cuda')

all_matches = []
start_ind = 0
while start_ind < len(zap_imgs):

    end_ind = np.min([len(zap_imgs), start_ind + 5000])
    dists = cdist(e2s_features_t[start_ind:end_ind], zap_features_t)
    zap_matches = list(np.argmin(dists.cpu().numpy(), axis=1))
    all_matches.extend(zap_matches)
    start_ind += 5000

all_matches = np.array(all_matches)

# Match between images from different datasets

e2s_zap_labels = final_labels.iloc[all_matches, :]
e2s_zap_labels = pd.DataFrame(np.repeat(e2s_zap_labels.values, 2, axis=0), columns=e2s_zap_labels.columns)
e2s_zap_labels__strings = final_labels__strings.iloc[all_matches, :]
e2s_zap_labels__strings = pd.DataFrame(np.repeat(e2s_zap_labels__strings.values, 2, axis=0),
                                       columns=e2s_zap_labels__strings.columns)


# Enlarge Lines And Reduce To x64

black_lines_kernel = np.ones((3, 3), np.uint8)


def inflate_black_lines(img):
    return 255. - cv2.dilate(255. - img, black_lines_kernel, iterations=2)


def convert_to_64(img):
    img = inflate_black_lines(img)
    return cv2.resize(img, (64, 64))


final_imgs = np.array([convert_to_64(x) for x in enumerate(e2s_imgs)])

# splits to train and test

train_indxs, test_indxs = train_test_split(np.arange(len(final_imgs)), test_size=0.1)

for indxs, mode in zip([train_indxs, test_indxs], ['train', 'test']):
    mode_imgs = final_imgs[indxs]
    mode_contents = e2s_zap_labels.iloc[indxs, :].values
    mode_contents_strings = e2s_zap_labels__strings.iloc[indxs, :].values
    mode_classes = classes[indxs]
    np.savez(join(save_folder, f'edges2shoes_x64_{mode}.npz'), imgs=mode_imgs,
             classes=mode_classes, n_classes=n_classes,
             contents=mode_contents, contents_strings=mode_contents_strings)
