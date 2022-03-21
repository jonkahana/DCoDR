import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm as tqdm
import faiss


def get_retrieval_attrs_and_modulos(data_name, all_contents, landmarks_clause=False):
    modulo_sizes = None
    allowed_dists = None
    if 'celeba' in data_name and landmarks_clause:
        modulo_sizes = []
        allowed_dists = {f'{i}': 0 for i in range(136)}
    elif data_name in ['smallnorb_train', 'smallnorb_test']:
        modulo_sizes = ['azimuth']
        allowed_dists = {'azimuth': 2, 'elevation': 0, 'lighting': 0}
    elif 'shapes3d' in data_name:
        modulo_sizes = []
        allowed_dists = {'floor_hue': 0, 'wall_hue': 0, 'object_hue': 0, 'scale': 0, 'shape': 0, 'orientation': 0}
        for cont in allowed_dists.keys():
            if cont in data_name:
                allowed_dists.pop(cont)
                break
    elif 'edges2shoes_x64' in data_name:
        modulo_sizes = []
        allowed_dists = {'shoe_type': 0, 'gender': 0}
    else:
        modulo_sizes = []
        allowed_dists = {'content': 0}

    modulo_sizes_d = {}
    for k in modulo_sizes:
        modulo_sizes_d[k] = len(np.unique(all_contents[k]))

    return allowed_dists, modulo_sizes_d


def check_match(a_conts: pd.Series, b_conts: pd.Series, allowed_dists: dict, modulo_sizes: dict):
    all_match = True
    attr_matches = dict()
    for k in allowed_dists.keys():
        diff = np.abs(a_conts[k] - b_conts[k])
        dist = min(diff, modulo_sizes[k] - diff) if k in modulo_sizes.keys() else diff
        is_attr_match = dist <= allowed_dists[k]
        attr_matches[k] = is_attr_match
        if not is_attr_match:
            all_match = False
    attr_matches['all'] = all_match
    return attr_matches


def check_k_matches(a_conts: pd.Series, b_conts: pd.DataFrame, allowed_dists: dict, modulo_sizes: dict):
    matches = []
    for i in range(len(b_conts)):
        matches.append(check_match(a_conts, b_conts.iloc[i, :], allowed_dists, modulo_sizes))
    matches = pd.DataFrame(matches)
    return matches


def get_retrieval_accuracies__from_all(args, all_contents, all_codes,
                                       test_contents, test_codes, test_class_ids,
                                       conts_allowed_dists: dict, conts_modulo_sizes: dict,
                                       k_neighbors=10, dist='l2'):
    accs = {k: np.ones(k_neighbors) for k in list(conts_allowed_dists.keys()) + ['all']}

    torch_test_codes = torch.tensor(test_codes.values).to(args.device)

    cls_counter = 0

    matches = {k: np.zeros((len(test_class_ids), k_neighbors))
               for k in list(conts_allowed_dists.keys()) + ['all']}

    torch_all_codes = torch.tensor(all_codes.values).to(args.device)

    if 'l2' in dist:
        dists = torch.cdist(torch_test_codes, torch_all_codes).cpu().numpy()
    elif dist == 'faiss':
        res = faiss.StandardGpuResources()
        faiss_index = faiss.IndexFlatL2(all_codes.shape[1])
        faiss_gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        faiss_gpu_index.add(all_codes.values.astype('float32'))
        dists, faiss_neighbors_indexes = faiss_gpu_index.search(test_codes.values.astype('float32'), k_neighbors + 5)
    else:
        raise ValueError(f'no such supported dist metric as {dist}')

    num_skipped = 0
    for i in range(len(test_class_ids)):

        if 'l2' in dist:
            cur_dists = dists[i, :]
            neighbors_indxs = np.argsort(cur_dists)[1:k_neighbors + 1]
        elif dist == 'faiss':
            neighbors_indxs = faiss_neighbors_indexes[i, 1:k_neighbors + 1]
        else:
            raise ValueError(f'no such supported dist metric as {dist}')

        attrs_matches = check_k_matches(test_contents.iloc[i, :],
                                        all_contents.iloc[neighbors_indxs, :],
                                        allowed_dists=conts_allowed_dists,
                                        modulo_sizes=conts_modulo_sizes)
        for k in matches.keys():
            matches[k][i - num_skipped, :len(neighbors_indxs)] = attrs_matches[k].values

    for k in matches.keys():
        cur_matches = np.array(matches[k])
        for top_n in range(1, k_neighbors + 1):
            top_n_matches = cur_matches[:, :top_n]
            top_n_acc = np.mean(np.sum(top_n_matches, axis=1) >= 1)
            accs[k][top_n - 1] = top_n_acc

    cls_counter += 1

    mean_accs = {}
    for k in accs.keys():

        mean_accuracy = np.mean(accs[k][0])
        print(f'{k} mean prediction accuracy:', mean_accuracy)
        mean_accs[f'{k}__mean_accuracy'] = mean_accuracy

    return mean_accs


def get_retrieval_figs(args, all_codes, all_class_ids, all_imgs,
                       test_codes, test_class_ids, test_imgs,
                       exp_name, chosen_epoch,
                       k_neighbors=5, dist='l2',
                       num_samples=4, save_results_path='.'):

    torch_test_codes = torch.tensor(test_codes.values).to(args.device)

    compared_indxs = np.arange(len(all_class_ids))
    compared_codes = torch.tensor(all_codes.iloc[compared_indxs, :].values).to(args.device)

    relevant_indxs = np.arange(len(test_class_ids))
    selected_is = np.random.choice(relevant_indxs, size=num_samples, replace=False)
    selected_cls_ids = [test_class_ids[i] for i in selected_is]

    if 'l2' in dist:
        rand_cls_dists = torch.cdist(torch_test_codes[selected_is], compared_codes).cpu().numpy()
    else:
        raise ValueError(f'no such supported dist metric as {dist}')

    fig = make_subplots(rows=num_samples, cols=k_neighbors + 1,
                        horizontal_spacing=0., vertical_spacing=0.03)

    figs_counter = 0
    for loop_index, (i, cls_id) in enumerate(zip(selected_is, selected_cls_ids)):

        img = (test_imgs[i] * 255.).astype(np.uint8)
        img = np.concatenate([img] * 3, axis=-1) if img.shape[-1] == 1 else img
        fig.add_trace(go.Image(z=img), row=loop_index + 1, col=1)
        figs_counter += 1

        skip_self = 1
        if 'l2' in dist:
            cur_dists = rand_cls_dists[loop_index, :]
            neighbors_indxs = compared_indxs[np.argsort(cur_dists)[skip_self:k_neighbors + skip_self]]
            neighbors_dists = np.sort(cur_dists)[skip_self:k_neighbors + skip_self]
        else:
            raise ValueError(f'no such supported dist metric as {dist}')

        for j, (neighbor_ind, neighbor_dist) in enumerate(zip(neighbors_indxs, neighbors_dists)):
            img = (all_imgs[neighbor_ind] * 255.).astype(np.uint8)
            img = np.concatenate([img] * 3, axis=-1) if img.shape[-1] == 1 else img
            fig.add_trace(go.Image(z=img), row=loop_index + 1, col=j + 2)
            figs_counter += 1

    fig.add_vline(x=test_imgs[0].shape[1] + 5, y0=0, y1=1, col=1, line_width=5)
    fig.update_xaxes(showticklabels=False,
                     title_standoff=0)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(width=1400, height=1000,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

    save_dir = os.path.join(save_results_path, 'figures')
    new_exp_name = exp_name.split('/')[-1] if '/' in exp_name else exp_name
    save_path = os.path.join(save_dir, f'{new_exp_name}__epoch_{chosen_epoch}__retrieval.png')
    os.makedirs(save_dir, exist_ok=True)
    fig.write_image(save_path)

    return fig

