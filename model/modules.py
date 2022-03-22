import os.path
from abc import ABC, abstractmethod
from typing import List, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
from os.path import join

import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn
from torchvision import models


def init_uniform_minus1_1(model):
    for param in model.parameters():
        if isinstance(param, nn.Linear) or isinstance(param, nn.Conv2d):
            nn.init.uniform_(param, -1., 1.)
    return


class Generator_Only_Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
        self.class_embedding = torch.nn.DataParallel(self.class_embedding)
        self.modulation = Modulation(config['class_dim'])
        self.modulation = torch.nn.DataParallel(self.modulation)

        self.generator = Generator(config['content_dim'], config['img_shape'])
        self.generator = torch.nn.DataParallel(self.generator)

    def forward(self, content_code, class_id=None):
        class_code = self.class_embedding(class_id)
        class_adain_params = self.modulation(class_code)
        generated_img = self.generator(content_code, class_adain_params)
        return {
            'img': generated_img,
        }

    def init_LO(self):
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class Encoder_Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        if 'use_pretrain' not in config:
            config['use_pretrain'] = True
        self.content_encoder = Moco_Resnet50(config['content_dim'], use_pretrain=config['use_pretrain'])
        self.content_encoder = torch.nn.DataParallel(self.content_encoder)

        if not 'use_fc_head' in self.config:
            self.config['use_fc_head'] = True

        if self.config['use_fc_head']:
            self.siamese = nn.Sequential(
                nn.Linear(in_features=config['content_dim'],
                          out_features=config['content_dim']),
                nn.LeakyReLU(),
                nn.Linear(in_features=config['content_dim'],
                          out_features=config['content_dim']),
                nn.LeakyReLU(),
                nn.Linear(in_features=config['content_dim'],
                          out_features=config['content_dim']),
            )
            self.siamese = torch.nn.DataParallel(self.siamese)

    def forward(self, img):

        z = self.content_encoder(img)
        if self.config['use_fc_head']:
            z = self.siamese(z)
        return {
            'code': z,
        }

    def init_LO(self):
        self.apply(self.weights_init_LO)

    @staticmethod
    def weights_init_LO(m):
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class Modulation(nn.Module):

    def __init__(self, code_dim, n_adain_layers=4, adain_dim=256):
        super().__init__()

        self.__n_adain_layers = n_adain_layers
        self.__adain_dim = adain_dim

        self.adain_per_layer = nn.ModuleList([
            nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
            for _ in range(n_adain_layers)
        ])

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

        return adain_params


class Generator(nn.Module):

    def __init__(self, content_dim, img_shape, n_adain_layers=4, adain_dim=256, use_AdaIN=True):
        super().__init__()

        self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim
        self.use_AdaIN = use_AdaIN

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
                out_features=self.__initial_height * self.__initial_width * adain_dim
            ),

            nn.LeakyReLU()
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(n_adain_layers):
            if self.use_AdaIN:
                last_layer = AdaptiveInstanceNorm2d(adain_layer_idx=i)
            else:
                last_layer = nn.BatchNorm2d(adain_dim)

            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
                nn.LeakyReLU(),
                last_layer
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

        self.last_conv_layers = [
            nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
            nn.LeakyReLU(),
        ]
        if img_shape[1] == 128:
            additional_layers = [nn.Conv2d(in_channels=64, out_channels=64, padding=2, kernel_size=5),
                                 nn.LeakyReLU(),
                                 ]
            self.last_conv_layers.extend(additional_layers)
        self.last_conv_layers.extend([
            nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
            nn.Sigmoid()
        ])
        self.last_conv_layers = nn.Sequential(*self.last_conv_layers)

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_adain_params=None):

        if self.use_AdaIN:
            if class_adain_params is None:
                raise ValueError(f'cant set self.use_AdaIN==True and not give AdaIN params')
            self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x


class Moco_Resnet50(nn.Module):

    def __init__(self, code_dim, use_pretrain=True,
                 pretrained_weights_path='pretrained_weights/moco_v2_800ep_pretrain.pth'):
        super().__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=128)
        self.resnet50.fc = nn.Sequential(nn.Linear(2048, 2048))
        if use_pretrain:
            if pretrained_weights_path is None or not os.path.exists(pretrained_weights_path):
                raise ValueError('pretrained weights location does not exist.\n' +
                                 'take weights from: https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar')
            state_d = torch.load(pretrained_weights_path)
            state_d['state_dict'] = {'.'.join(x.split('.')[2:]): state_d['state_dict'][x] for x in
                                     state_d['state_dict'].keys()}
            keys = list(state_d['state_dict'].keys())[:-2]
            state_d['state_dict'] = {k: state_d['state_dict'][k] for k in keys}  # remove last layer of dim. reduction
            self.resnet50.load_state_dict(state_dict=state_d['state_dict'])
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=2048, out_features=code_dim)
        )

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.tile((1, 3, 1, 1))
        x = self.resnet50(x)
        x = self.fc_layers(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, adain_layer_idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.adain_layer_idx = adain_layer_idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True
        )

        out = out.view(b, c, *x.shape[2:])
        return out


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class PerceptualDistance(nn.Module):

    def __init__(self, layer_ids=[2, 7, 12, 21, 30]):
        super().__init__()

        self.backbone = NetVGGFeatures(layer_ids)
        self.layer_ids = layer_ids

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.backbone(I1)
        f2 = self.backbone(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


def get_indxs_of_uniques(arr):
    """
    source: https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    """
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(arr)

    # sorts records array so all unique elements are together
    sorted_records_array = arr[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])
    res = {vals[i]: res[i] for i in range(len(vals))}

    return res


def get_rand_negs_sims(z_vecs, b_size, num_rand_negs, classes=None):
    if classes is None:
        rand_negs_indxs = np.array([np.random.choice(np.delete(np.arange(b_size), i), num_rand_negs, replace=False)
                                    for i in range(b_size)])
    else:
        allow_replace = False
        if pd.Series(classes).value_counts().min() - 1 < num_rand_negs:
            allow_replace = True
        classes_indxs = get_indxs_of_uniques(classes)
        # chooses num_rand_negs negative samples from the same class
        rand_negs_indxs = np.array(
            [np.random.choice(np.delete(classes_indxs[classes[i]], np.argwhere(classes_indxs[classes[i]] == i)),
                              num_rand_negs, replace=allow_replace) for i in range(b_size)])
    dim0_indxs = torch.arange(b_size).tile((num_rand_negs, 1)).t().flatten()
    dim1_indxs = torch.tensor(rand_negs_indxs).flatten()

    # rand_negs_norms = torch.matmul(z_vecs, z_vecs.T)  # + addition
    # rand_negs_norms = rand_negs_norms[dim0_indxs, dim1_indxs].view(b_size, num_rand_negs)
    rand_negs_norms = torch.sum(z_vecs[dim0_indxs] * z_vecs[dim1_indxs], dim=1).view(b_size, num_rand_negs)

    return rand_negs_norms


def get_pos_sims(z_vecs, posneg_zs):
    return torch.matmul(posneg_zs, z_vecs.unsqueeze(-1)).squeeze(-1)  # + addition


class SimCLR_Loss_w_Pos(nn.Module):
    """Extended InfoNCE objective for normalized representations based on an cosine similarity.

    Args:
        num_rand_negs: number of random negatives to add for each z vector
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
            self,
            num_rand_negs: int,
            tau: float = 1.0,
            alpha: float = 0.5,
            device='cuda',
            num_pos=1
    ):
        super(SimCLR_Loss_w_Pos, self).__init__()

        self.tau = tau
        self.alpha = alpha

        self.num_pos = num_pos
        self.num_rand_negs = num_rand_negs

        self.device = device

    def __call__(self, z_vecs, pos_z_vecs, classes=None, normalize=True):

        if normalize:
            z_vecs = F.normalize(z_vecs, dim=-1)
            pos_z_vecs = F.normalize(pos_z_vecs, dim=-1)

        batch_size = z_vecs.shape[0]

        if classes is None:
            num_rand_neg = np.min([batch_size - 1, self.num_rand_negs])
        else:
            num_samples_per_class = batch_size // len(np.unique(classes))
            num_rand_neg = np.min([num_samples_per_class - 1, self.num_rand_negs])
        if num_rand_neg != 0:
            rand_negs_sims = get_rand_negs_sims(z_vecs, batch_size, num_rand_neg,
                                                classes=classes).to(self.device)
        else:
            raise ValueError('cant have no negs in the CLL! Check num_rand_negs parameter is not 0')

        pos_sims = get_pos_sims(z_vecs, pos_z_vecs).to(self.device)
        if self.num_pos == 0:
            pos_sims = torch.ones_like(pos_sims).to(self.device).double()

        neg_and_pos_sims = torch.cat((rand_negs_sims, pos_sims), dim=-1)
        loss_pos = torch.logsumexp(pos_sims / self.tau, dim=-1)
        loss_neg = torch.logsumexp(neg_and_pos_sims / self.tau, dim=-1)

        loss = -2 * (self.alpha * loss_pos - (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)

        return loss_mean
