import gc
import os
import itertools
import pickle
from tqdm import tqdm
from os.path import join
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import *
from model.utils import *
from dataset import Images_Augmentation_Subset


def get_orig_dataset(classes, config, imgs):
    if config['dataset_name'] in ['celeba_x64_train', 'celeba_x64_test']:
        orig_imgs_ids = np.arange(imgs.shape[0])
        num_samples_by_class = pd.Series(classes).value_counts()
        small_classes = list(num_samples_by_class[num_samples_by_class <= 19].index)
        keep_indxs = np.where(~pd.Series(classes).isin(small_classes))[0]
        imgs = imgs[keep_indxs]
        imgs_ids = orig_imgs_ids[keep_indxs]
        classes = classes[keep_indxs]
    else:
        imgs_ids = np.arange(imgs.shape[0])
    data = dict(
        img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
        img_id=torch.from_numpy(imgs_ids),
        class_id=torch.from_numpy(classes.astype(np.int64))
    )
    orig_dataset = NamedTensorDataset(data)

    return orig_dataset


def prep_data_loader(class_negs, classes, config, dataset):
    b_sampler = None
    is_shuffle = True
    drop_last = True
    b_size = config['train']['batch_size']
    if class_negs:
        is_shuffle = False
        drop_last = False
        b_sampler = BalancedBatchSampler(classes.astype(np.int64),
                                         config['train']['num_b_cls'],
                                         config['train']['num_b_cls_samp'])
        b_size = 1  # the default value of batch size in torch DataLoader
    data_loader = DataLoader(
        dataset, batch_size=b_size,
        shuffle=is_shuffle, batch_sampler=b_sampler,
        num_workers=config['train']['workers'], pin_memory=True, drop_last=drop_last
    )
    return data_loader


class Trainer:

    def __init__(self, config=None):
        super().__init__()

        self.config = config
        if config is None or 'device' not in self.config.keys():
            self.config = {'device': 'cuda'}
        self.device = self.config['device']
        self.generative_model = None
        self.encoder_model = None

    def load(self, model_dir, generative=False, encoder=False, epoch='', load_config=True):
        if load_config:
            with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
                self.config = pickle.load(config_fd)

        if generative:
            self.generative_model.load_state_dict(torch.load(os.path.join(model_dir, f'generator_{epoch}.pth')))

        if encoder:
            if self.encoder_model is None:
                self.encoder_model = Encoder_Model(self.config)
            self.encoder_model.load_state_dict(
                torch.load(os.path.join(model_dir, f'encoder_{epoch}.pth')))

    def save(self, model_dir, generative=False, discriminative=False, epoch=''):
        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
            config_copy = deepcopy(self.config)
            if 'dataset_args' in config_copy:
                config_copy['dataset_args'].pop('pos_augments')
            pickle.dump(config_copy, config_fd)

        if generative:
            torch.save(self.generative_model.state_dict(), os.path.join(model_dir, f'generator_{epoch}.pth'))

        if discriminative:
            torch.save(self.encoder_model.state_dict(), os.path.join(model_dir, f'encoder_{epoch}.pth'))

    def generate_samples(self, dataset, n_samples=5, randomized=False):

        self.encoder_model.eval()
        self.generative_model.eval()

        if randomized:
            random = np.random
        else:
            random = np.random.RandomState(seed=1234)

        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

        samples = dataset[img_idx]
        samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
        with torch.no_grad():
            samples['encoder_code'] = F.normalize(self.encoder_model(samples['img'])['code'].detach())

        blank = torch.ones_like(samples['img'][0])
        output = [torch.cat([blank] + list(samples['img']), dim=2)]
        for i in range(n_samples):

            converted_imgs = [samples['img'][i]]

            for j in range(n_samples):
                with torch.no_grad():
                    out = self.generative_model(samples['encoder_code'][[j]], samples['class_id'][[i]])
                converted_imgs.append(out['img'][0])

            output.append(torch.cat(converted_imgs, dim=2))

        return torch.cat(output, dim=1)

    def train_DCoDR_norec(self, imgs, classes, model_dir, tensorboard_dir, class_negs=True, test_data=None):

        self.encoder_model = Encoder_Model(self.config)

        orig_dataset = get_orig_dataset(classes, self.config, imgs)
        augmentations_dataset = Images_Augmentation_Subset(orig_dataset, **self.config['dataset_args'])
        data_loader = prep_data_loader(class_negs, orig_dataset.named_tensors['class_id'].numpy(),
                                       self.config, augmentations_dataset)

        test_loader = None
        if test_data is not None:
            test_orig_set = get_orig_dataset(test_data['classes'], self.config, test_data['imgs'])
            augmentations_test_set = Images_Augmentation_Subset(test_orig_set, **self.config['dataset_args'])
            test_loader = prep_data_loader(class_negs, test_data['classes'], self.config, augmentations_test_set)

        if self.config['load_weights']:
            self.load(self.config['load_weights_exp'], encoder=True,
                      epoch=str(self.config['load_weights_epoch']), load_config=False)
        else:
            self.encoder_model.init_LO()
        self.encoder_model.to(self.device)

        criterion = SimCLR_Loss_w_Pos(num_rand_negs=self.config['num_rand_negs'],
                                      tau=self.config['tau'],
                                      num_pos=self.config['num_pos']).to(self.device)

        encoder_params = [self.encoder_model.content_encoder.parameters()]
        if self.config['use_fc_head']:
            encoder_params.append(self.encoder_model.siamese.parameters())
        optim_params = itertools.chain(*encoder_params)
        optimizer = Adam([
            {
                'params': optim_params,
                'lr': self.config['train']['learning_rate']['encoder']
            }
        ], betas=(0.5, 0.999))

        scheduler = StepLR(optimizer, step_size=30 * len(data_loader), gamma=0.33)

        summary = SummaryWriter(log_dir=tensorboard_dir)

        def prep_outs(batch):

            batch_size = batch['img'].shape[0]
            pos_zs_shape = (batch_size, batch['pos_imgs'].shape[1], -1)

            orig_out = self.encoder_model(batch['img'])
            pos_out = self.encoder_model(torch.cat([x for x in batch['pos_imgs']], dim=0))
            for k in pos_out.keys():
                pos_out[k] = pos_out[k].view(pos_zs_shape)

            b_classes = batch['class_id'].detach().cpu().numpy() if class_negs else None

            return orig_out, pos_out, b_classes

        def calc_loss(orig_out, pos_out, b_classes, b_counter, train_test='train'):

            orig_codes = orig_out['code']
            pos_codes = pos_out['code']

            loss = criterion(orig_codes, pos_codes, classes=b_classes)
            summary.add_scalar(tag=f'{train_test}/cl_loss', scalar_value=loss.item(), global_step=b_counter)

            return loss

        train_loss, test_loss = AverageMeter(), AverageMeter()
        b_counter, test_b_counter = 0, 0
        for epoch in range(self.config['train']['n_epochs']):
            self.encoder_model.train()
            train_loss.reset()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                orig_out, pos_out, b_classes = prep_outs(batch)
                loss = calc_loss(orig_out, pos_out, b_classes, b_counter)

                summary.add_scalar(tag=f'train/batch_loss', scalar_value=loss.item(), global_step=b_counter)
                summary.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss.update(loss.item())
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(loss=train_loss.avg)
                b_counter += 1

            pbar.close()
            summary.add_scalar(tag=f'train/epoch_loss', scalar_value=train_loss.avg, global_step=epoch)

            if epoch % self.config['save_every'] == 0:
                self.save(model_dir, generative=False, discriminative=True, epoch=str(epoch))

            if epoch % self.config['eval_every'] == 0 and test_loader is not None:
                self.encoder_model.eval()
                test_loss.reset()
                pbar = tqdm(iterable=test_loader)
                for batch in pbar:
                    batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                    with torch.no_grad():
                        orig_out, pos_out, b_classes = prep_outs(batch)
                        loss = calc_loss(orig_out, pos_out, b_classes, test_b_counter, train_test='test')

                    summary.add_scalar(tag=f'test\batch_loss', scalar_value=loss.item(), global_step=test_b_counter)
                    summary.flush()

                    test_loss.update(loss.item())
                    pbar.set_description_str('|| TEST || epoch #{}'.format(epoch))
                    pbar.set_postfix(loss=test_loss.avg)
                    test_b_counter += 1

                pbar.close()
                summary.add_scalar(tag=f'test\epoch_loss', scalar_value=test_loss.avg, global_step=epoch)

        summary.close()

    def train_DCoDR(self, imgs, classes, model_dir, tensorboard_dir, class_negs=True, test_data=None):

        # region define models - gen + enc
        self.encoder_model = Encoder_Model(self.config)
        self.generative_model = Generator_Only_Model(self.config)
        # endregion

        # region prepare datasets
        orig_dataset = get_orig_dataset(classes, self.config, imgs)
        augmentations_dataset = Images_Augmentation_Subset(orig_dataset, **self.config['dataset_args'])
        data_loader = prep_data_loader(class_negs, orig_dataset.named_tensors['class_id'].numpy(),
                                       self.config, augmentations_dataset)

        test_loader = None
        if test_data is not None:
            test_orig_set = get_orig_dataset(test_data['classes'], self.config, test_data['imgs'])
            augmentations_test_set = Images_Augmentation_Subset(test_orig_set, **self.config['dataset_args'])
            test_loader = prep_data_loader(class_negs, test_data['classes'], self.config, augmentations_test_set)
        # endregion prepare datasets

        # region init. models

        if self.config['load_weights']:
            self.load(self.config['load_weights_exp'], encoder=True, generative=True,
                      epoch=str(self.config['load_weights_epoch']), load_config=False)
        else:
            self.encoder_model.init_LO()
            self.generative_model.init_LO()
        self.encoder_model.to(self.device)
        self.generative_model.to(self.device)

        # endregion init. models

        # region criterions + optimizers + tensorboard
        gen_criterion = PerceptualDistance().to(self.device)
        criterion = SimCLR_Loss_w_Pos(num_rand_negs=self.config['num_rand_negs'],
                                      tau=self.config['tau'],
                                      num_pos=self.config['num_pos']).to(self.device)

        encoder_params = [self.encoder_model.content_encoder.parameters()]
        if self.config['use_fc_head']:
            encoder_params.append(self.encoder_model.siamese.parameters())
        enc_optim_params = {
            'params': itertools.chain(*encoder_params),
            'lr': self.config['train']['learning_rate']['encoder']
        }

        gen_optim_params = [{
            'params': itertools.chain(self.generative_model.modulation.parameters(),
                                      self.generative_model.generator.parameters()),
            'lr': self.config['train']['learning_rate']['generator']
        }, {
            'params': itertools.chain(self.generative_model.class_embedding.parameters()),
            'lr': self.config['train']['learning_rate']['latent']
        }]

        optim_params = [enc_optim_params]
        optim_params.extend(gen_optim_params)

        optimizer = Adam(optim_params, betas=(0.5, 0.999))

        scheduler = StepLR(optimizer, step_size=30 * len(data_loader), gamma=0.33)

        summary = SummaryWriter(log_dir=tensorboard_dir)

        # endregion criterions + optimizers + tensorboard

        def prep_outs(batch):

            batch_size = batch['img'].shape[0]
            pos_zs_shape = (batch_size, batch['pos_imgs'].shape[1], -1)

            orig_out = self.encoder_model(batch['img'])
            pos_out = self.encoder_model(torch.cat([x for x in batch['pos_imgs']], dim=0))
            for k in pos_out.keys():
                pos_out[k] = pos_out[k].view(pos_zs_shape)

            b_classes = batch['class_id'].detach().cpu().numpy() if class_negs else None

            gen_out = self.generative_model(F.normalize(orig_out['code']), batch['class_id'].detach())
            if gen_out['img'].shape[1] == 1:
                gen_out['img'] = torch.cat([gen_out['img']] * 3, dim=1)
                batch['img'] = torch.cat([batch['img']] * 3, dim=1)

            return orig_out, pos_out, b_classes, gen_out

        def calc_loss(orig_out, pos_out, b_classes, b_counter, gen_out, train_test='train'):

            orig_codes = orig_out['code']
            pos_codes = pos_out['code']

            loss = criterion(orig_codes, pos_codes, classes=b_classes)

            if gen_out['img'].shape[1] == 1:
                gen_out['img'] = torch.cat([gen_out['img']] * 3, dim=1)
                batch['img'] = torch.cat([batch['img']] * 3, dim=1)
            reconstruction_loss = gen_criterion(gen_out['img'], batch['img'])
            loss += self.config['reconstruction_decay'] * reconstruction_loss
            summary.add_scalar(tag=f'{train_test}/reconstruction_loss',
                               scalar_value=reconstruction_loss.item(),
                               global_step=b_counter)

            return loss

        train_loss, test_loss = AverageMeter(), AverageMeter()
        b_counter, test_b_counter = 0, 0
        for epoch in range(self.config['train']['n_epochs']):

            self.encoder_model.train()
            train_loss.reset()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                orig_out, pos_out, b_classes, gen_out = prep_outs(batch)
                loss = calc_loss(orig_out, pos_out, b_classes, b_counter, gen_out=gen_out)

                summary.add_scalar(tag=f'train/batch_loss', scalar_value=loss.item(), global_step=b_counter)
                summary.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss.update(loss.item())
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(loss=train_loss.avg)
                b_counter += 1

            pbar.close()
            summary.add_scalar(tag=f'train/epoch_loss', scalar_value=train_loss.avg, global_step=epoch)

            if epoch % self.config['save_every'] == 0:
                self.save(model_dir, generative=True, discriminative=True, epoch=str(epoch))

            if epoch % self.config['eval_every'] == 0 and test_loader is not None:
                self.encoder_model.eval()
                test_loss.reset()
                pbar = tqdm(iterable=test_loader)
                for batch in pbar:
                    batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                    with torch.no_grad():
                        orig_out, pos_out, b_classes, gen_out = prep_outs(batch)
                        loss = calc_loss(orig_out, pos_out, b_classes, test_b_counter,
                                         gen_out=gen_out, train_test='test')

                    summary.add_scalar(tag=f'test\batch_loss', scalar_value=loss.item(), global_step=test_b_counter)
                    summary.flush()

                    test_loss.update(loss.item())
                    pbar.set_description_str('|| TEST || epoch #{}'.format(epoch))
                    pbar.set_postfix(loss=test_loss.avg)
                    test_b_counter += 1

                pbar.close()
                summary.add_scalar(tag=f'test\epoch_loss', scalar_value=test_loss.avg, global_step=epoch)

            if epoch % 5 == 0:
                with torch.no_grad():
                    fixed_sample_img = self.generate_samples(orig_dataset, n_samples=5, randomized=False)
                    random_sample_img = self.generate_samples(orig_dataset, n_samples=5, randomized=True)

                summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
                summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

        summary.close()
