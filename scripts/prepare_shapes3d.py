from os.path import join as join
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os
import h5py
from copy import deepcopy

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}

dataset_path = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", "3dshapes.h5")
save_folder = 'cache/preprocess'
os.makedirs(save_folder, exist_ok=True)


save_reduce_version = True
num_reduce = 50000

def enumerate_cols(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    all_maps = {}
    for col in columns:
        mapping = {x: i for i, x in enumerate(sorted(np.unique(df[col])))}
        df[col] = df[col].apply(lambda x: mapping[x])
        all_maps[col] = mapping
    return df, all_maps

if __name__ == '__main__':

    # load dataset
    dataset = h5py.File(dataset_path, 'r')
    print(dataset.keys())
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    np_imgs = np.array(images)
    np_labels = np.array(labels)

    contents_df = pd.DataFrame(np_labels, columns=_FACTORS_IN_ORDER, index=np.arange(len(np_labels)))
    orig_contents = deepcopy(contents_df)

    mapping_dict = {x: i for i, x in enumerate(sorted(contents_df['shape'].unique()))}
    contents_df['shape'] = contents_df['shape'].apply(lambda x: mapping_dict[x])

    train_indxs, test_indxs = train_test_split(range(len(contents_df)), test_size=0.1,
                                               stratify=contents_df['shape'].values)

    classes = contents_df['shape'].values
    n_classes = len(np.unique(classes))
    for indxs, train_test in zip([train_indxs, test_indxs], ['np_imgs', 'test']):
        np.savez(join(save_folder, f'shapes3d__class_shape__{train_test}.npz'),
                 imgs=np_imgs[indxs],
                 contents=contents_df.drop('shape', axis=1).values[indxs],
                 classes=classes[indxs], n_classes=n_classes,
                 orig_contents=orig_contents.drop('shape', axis=1).values[indxs],
                 orig_classes=orig_contents['shape'].values[indxs])

    if save_reduce_version:

        str_num_train = str(num_reduce)[::-1].replace('000000', 'M').replace('000', 'K')[::-1]

        keep_indxs = list(np.random.choice(np.arange(len(train_indxs)), size=num_reduce, replace=False))
        keep_indxs = list(train_indxs[keep_indxs])

        np.savez(join(save_folder, f'shapes3d__class_shape__train__reduced_{str_num_train}.npz'),
                 imgs=np_imgs[keep_indxs],
                 contents=contents_df.drop('shape', axis=1).values[keep_indxs],
                 classes=classes[keep_indxs], n_classes=n_classes,
                 orig_contents=orig_contents.drop('shape', axis=1).values[keep_indxs],
                 orig_classes=orig_contents['shape'].values[keep_indxs])

