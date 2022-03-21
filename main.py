import argparse

import time
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import os
from os.path import join
import shutil
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from transforms_utils import AUGS_DICT

import dataset
from assets import AssetManager, DictX
from model.training import Trainer
from config import base_config
from parse_args import *

# This Code was Based on https://github.com/avivga/lord


# region globals

pos_transforms = [
    AUGS_DICT['gblurr'],
    AUGS_DICT['high_contrast'],
    AUGS_DICT['crop'],
    AUGS_DICT['high_satur']
]

# endregion globals


# region Helper Functions

def replace_pos_transforms(names_list):
    global pos_transforms
    new_transforms = []
    for name in names_list:
        new_transforms.append(AUGS_DICT[name])
    pos_transforms = new_transforms
    return


def save_experiment_hyper_params(args, model_dir, prefix='', verbose=True):
    with open(join(model_dir, f'{prefix}args.txt'), 'w+') as f:
        # print args to file
        f.write('\n\n\n')
        f.write('Experiment Args:\n\n')
        for k in args:
            f.write(f'\t {k}: {args[k]}\n')  # print config to file
        f.write('\n\n\n')
        f.write('Base Config:\n\n')
        for k in base_config:
            if type(base_config[k]) == dict:
                f.write(f'\n\t{k}:\n')
                for k_ in base_config[k]:
                    f.write(f'\t\t{k_}: {base_config[k][k_]}\n')
            else:
                f.write(f'\t{k}: {base_config[k]}\n')
        # print transforms to file
        f.write('\n\n\n')
        f.write('Pos Transforms:\n')
        if type(pos_transforms) == list:
            for t in pos_transforms:
                f.write(f'\t{t}\n')
        else:
            f.write(f'\t{pos_transforms}\n')
    shutil.copyfile('config.py', join(model_dir, 'base_config.py'))
    if verbose:
        with open(join(model_dir, f'{prefix}args.txt'), 'r') as f:
            for line in f:
                print(line)
    return


def load_np_data(assets, data_name, cuda=True):
    data = dict(np.load(assets.get_preprocess_file_path(data_name), allow_pickle=True))
    if 'n_classes' in data.keys() and not len(data['n_classes'].shape) > 0:
        data['n_classes'] = int(data['n_classes'])
    imgs = data['imgs'].astype(np.float32)
    imgs = imgs / 255.0
    data['imgs'] = imgs

    config = dict(
        img_shape=imgs.shape[1:],
        n_imgs=imgs.shape[0],
        device="cuda" if torch.cuda.is_available() and cuda else "cpu"
    )
    return data, imgs, config


def insert_args_to_config(config, args, args_type):
    if args_type == 'global':
        config['train']['workers'] = args.num_workers
        config['train']['batch_size'] = args.batch_size
        config['train']['n_epochs'] = args.epochs
        config['content_dim'] = args.content_dim
        config['class_dim'] = args.class_dim
        config['dataset_name'] = args.data_name
        config['enc_arch'] = args.enc_arch
        config['load_weights'] = args.load_weights
        config['load_weights_exp'] = args.load_weights_exp
        config['load_weights_epoch'] = args.load_weights_epoch
        config['use_pretrain'] = args.use_pretrain
    elif args_type == 'cl':
        config['train']['num_b_cls'] = args.num_b_cls
        config['train']['num_b_cls_samp'] = args.num_b_cls_samp
        config['num_rand_negs'] = args.num_rand_negs
        config['num_pos'] = args.num_pos
        config['tau'] = args.tau
        config['use_fc_head'] = args.use_fc_head
    else:
        raise ValueError(f'No such args_type as {args_type}')
    return config


def get_dataset_args_dict(args):
    dataset_args = dict(pos_augments=pos_transforms,
                        num_pos=args.num_pos,
                        )
    return dataset_args


def post_process_args(args):
    args.model_name = join(args.data_name, args.exp_name)
    if 'shifting_args' in args and args.shifting_args is not None:
        args.shifting_args = [float(x) for x in eval(args.shifting_args)]
    if 'used_transforms' in args and args.used_transforms is not None:
        args.used_transforms = eval(args.used_transforms)
    return args


# endregion Helper Functions


# region Experiment Functions


def train_multi_arg(args):
    if 'used_transforms' in args and args.used_transforms is not None:
        replace_pos_transforms(names_list=args.used_transforms)

    assets = AssetManager(args.base_dir)
    model_dir = assets.recreate_model_dir(args.model_name, keep_prev=True)
    tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name, keep_prev=True)
    save_experiment_hyper_params(args, model_dir, verbose=False)
    writer = SummaryWriter(tensorboard_dir)
    for i, shift_val in enumerate(args.shifting_args):
        if args.base_exp == 'DCoDR_norec':
            train_DCoDR_norec__pipeline(args, shifting_arg=shift_val, multi_arg_key=args.shifting_key)
        elif args.base_exp == 'DCoDR':
            train_DCoDR__pipeline(args, shifting_arg=shift_val, multi_arg_key=args.shifting_key)
        else:
            raise ValueError(f'Unsupported base experiment:{args.base_exp}, for multi args ablation')
    writer.flush()
    writer.close()
    return


def train_DCoDR_norec__pipeline(args, shifting_arg=None, multi_arg_key=None):
    if 'used_transforms' in args and args.used_transforms is not None and shifting_arg is None:
        replace_pos_transforms(names_list=args.used_transforms)

    assets = AssetManager(args.base_dir)
    if shifting_arg is not None:
        model_dir = assets.get_model_dir(args.model_name)
        tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
        model_dir = join(model_dir, f'{multi_arg_key}_' + str(shifting_arg).replace('.', '_'))
        tensorboard_dir = join(tensorboard_dir, f'{multi_arg_key}_' + str(shifting_arg).replace('.', '_'))
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        prefix = ''
    else:
        model_dir = assets.recreate_model_dir(args.model_name)
        tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)
        prefix = ''
    if shifting_arg is not None:
        args[multi_arg_key] = shifting_arg

    data, imgs, config = load_np_data(assets, args.data_name, args.cuda)
    test_data = None
    if args.test_data_name is not None:
        test_data, test_imgs, _ = load_np_data(assets, args.test_data_name, args.cuda)
        test_data['imgs'] = test_imgs

    config.update(base_config)
    if shifting_arg is not None:
        # do it  a second time in case the argument was in base_config
        args[multi_arg_key] = shifting_arg
    config = insert_args_to_config(config, args, 'global')
    config = insert_args_to_config(config, args, 'cl')
    config['dataset_args'] = get_dataset_args_dict(args)

    save_experiment_hyper_params(args, model_dir, prefix=prefix)

    trainer = Trainer(config)

    trainer.train_DCoDR_norec(
        imgs=imgs,
        classes=data['classes'],
        model_dir=model_dir,
        tensorboard_dir=tensorboard_dir,
        class_negs=args.class_negs,
        test_data=test_data
    )
    trainer.save(model_dir, generative=False, discriminative=True, epoch='last')


def train_DCoDR__pipeline(args, shifting_arg=None, multi_arg_key=None):
    if 'used_transforms' in args and args.used_transforms is not None and shifting_arg is None:
        replace_pos_transforms(names_list=args.used_transforms)

    assets = AssetManager(args.base_dir)
    if shifting_arg is not None:
        model_dir = assets.get_model_dir(args.model_name)
        tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
        model_dir = join(model_dir, f'{multi_arg_key}_' + str(shifting_arg).replace('.', '_'))
        tensorboard_dir = join(tensorboard_dir, f'{multi_arg_key}_' + str(shifting_arg).replace('.', '_'))
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        prefix = ''
    else:
        model_dir = assets.recreate_model_dir(args.model_name)
        tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)
        prefix = ''

    if shifting_arg is not None:
        args[multi_arg_key] = shifting_arg

    data, imgs, config = load_np_data(assets, args.data_name, args.cuda)
    config['n_classes'] = data['n_classes']

    test_data = None
    if args.test_data_name is not None:
        test_data, test_imgs, _ = load_np_data(assets, args.test_data_name, args.cuda)
        test_data['imgs'] = test_imgs

    config.update(base_config)
    if shifting_arg is not None:
        # do it  a second time in case the arguement was in base_config
        args[multi_arg_key] = shifting_arg
    config = insert_args_to_config(config, args, 'global')
    config = insert_args_to_config(config, args, 'cl')

    config['reconstruction_decay'] = args.reconstruction_decay
    if args.gen_lr is not None:
        config['train']['learning_rate']['generator'] = args.gen_lr
        config['train']['learning_rate']['encoder'] = args.gen_lr

    config['dataset_args'] = get_dataset_args_dict(args)

    save_experiment_hyper_params(args, model_dir, prefix=prefix)

    trainer = Trainer(config)

    trainer.train_DCoDR(
        imgs=imgs,
        classes=data['classes'],
        model_dir=model_dir,
        tensorboard_dir=tensorboard_dir,
        class_negs=args.class_negs,
        test_data=test_data
    )
    trainer.save(model_dir, generative=True, discriminative=True, epoch='last')


# endregion Experiment Functions


def parse_main_args():
    parser = argparse.ArgumentParser()
    parser = get_general_args(parser)

    exp_type_parsers = parser.add_subparsers(dest='exp-specific-params')

    # region experiment type parameters parsing

    # region DCoDR_norec_multi_arg
    argparser__DCoDR_norec = exp_type_parsers.add_parser('DCoDR_norec_multi_arg')
    argparser__DCoDR_norec = get_basic_cl_args(argparser__DCoDR_norec)
    argparser__DCoDR_norec.add_argument('--shifting-args', type=str, required=True)
    argparser__DCoDR_norec.add_argument('--shifting-key', type=str, required=True)
    argparser__DCoDR_norec.add_argument('--base-exp', type=str, default='DCoDR_norec')
    argparser__DCoDR_norec.set_defaults(func=train_multi_arg)
    # endregion

    # region DCoDR_multi_arg
    argparser__DCoDR = exp_type_parsers.add_parser('DCoDR_multi_arg')
    argparser__DCoDR = get_basic_cl_args(argparser__DCoDR)
    argparser__DCoDR.add_argument('--gen-lr', type=float, default=None)
    argparser__DCoDR.add_argument('--reconstruction-decay', type=float, default=0.3)
    argparser__DCoDR.add_argument('--shifting-args', type=str, required=True)
    argparser__DCoDR.add_argument('--shifting-key', type=str, required=True)
    argparser__DCoDR.add_argument('--base-exp', type=str, default='DCoDR')
    argparser__DCoDR.set_defaults(func=train_multi_arg)
    # endregion

    # endregion experiment type parameters

    return parser


def main():

    parser = parse_main_args()
    args = parser.parse_args()
    args = post_process_args(args)
    args = DictX(vars(args))
    args.func(args)


if __name__ == '__main__':
    main()
