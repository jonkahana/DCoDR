import torch.utils.data
import torchvision.models

from model.training import *
from main import *

import numpy as np
import pandas as pd

import os
from os.path import join
import argparse
import gc
from copy import deepcopy
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from retrieval_utils import get_retrieval_accuracies__from_all, get_retrieval_figs, get_retrieval_attrs_and_modulos

np.random.seed(2)

save_results_path = 'results'
os.makedirs(save_results_path, exist_ok=True)

weights_folder = 'eval_weights/tmp_weights'

max_unique_vals_for_classification = 200000


# region Evaluation Args

def get_evaluated_experiments(args):
    """
    searches and fetches experiment names inside the given folders in evaluated-exp-names / root-exps
    """
    exps_data = args.model_train_data if args.model_train_data is not None else args.train_data_name
    for root_exp_name in args.root_exps:
        root_exp_path = join(args.base_dir, 'models', exps_data, root_exp_name)
        for obj in os.listdir(root_exp_path)[::-1]:
            obj_path = join(root_exp_path, obj)
            if os.path.isdir(obj_path):
                skip = True
                for x in os.listdir(obj_path):
                    if np.sum([str(args.chosen_epoch[i]) in x for i in range(len(args.chosen_epoch))]) > 0:
                        skip = False
                        break
                if skip:
                    continue
                args.evaluated_exp_names.append(join(root_exp_name, obj))
    print('\n'.join(args.evaluated_exp_names))

    return args.evaluated_exp_names


def get_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, required=True)
    parser.add_argument('--eval-type', type=str, default='prediction', help="prediction || retrieval")
    parser.add_argument('--eval-name', type=str, required=True)
    parser.add_argument('--evaluated-exp-names', type=str, required=True,
                        help="list of paths to directories of experiments")
    parser.add_argument('--root-exps', type=str, default='[]', help='optional in addition to --evaluated-exp-names. '
                                                                    'a list of given folders, which containing '
                                                                    'folders for experiments.')
    parser.add_argument('--train-data-name', type=str, required=True,
                        help="""
                        options are: 
                            || cars3d_train || cars3d_test ||
                            || smallnorb_train || smallnorb_test ||
                            || celeba_x64_train || celeba_x64_test ||
                            || edges2shoes_x64_train || edges2shoes_x64_test ||
                            || shapes3d__class_shape__train || shapes3d__class_shape__test ||  
                        """)
    parser.add_argument('--test-data-name', type=str, default=None)
    parser.add_argument('--delete-weights-folder', type=bool, default=False)
    parser.add_argument("--chosen-epoch", type=str, default='last', help='epoch num or the striong last for last epoch.'
                                                                         'can accept a list of epochs to check as well')
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument('--retrieval-dist', type=str, default='l2', help='l2 || faiss')
    parser.add_argument('--model-train-data', type=str, default=None, help="dataset name of the trained model")
    args = parser.parse_args()

    args.evaluated_exp_names = eval(args.evaluated_exp_names)
    args.root_exps = eval(args.root_exps)
    if args.test_data_name is None:
        args.test_data_name = args.train_data_name.replace('train', 'test')

    if args.chosen_epoch == 'last':
        args.chosen_epoch = ['last']
    else:
        args.chosen_epoch = eval(str(args.chosen_epoch))
    if type(args.chosen_epoch) == str or type(args.chosen_epoch) == int:
        args.chosen_epoch = [args.chosen_epoch]

    args.evaluated_exp_names = get_evaluated_experiments(args)

    return args


# endregion Evaluation Args


# region DataSet Related Helpers

def get_content_colnames(args):
    if 'celeba' in args.train_data_name:
        content_cols = list(np.arange(136).astype(str))
    elif args.train_data_name in ['smallnorb_train', 'smallnorb_test']:
        content_cols = ['azimuth', 'elevation', 'lighting']
    elif 'shapes3d' in args.train_data_name:
        content_cols = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        for cont in content_cols:
            if cont in args.train_data_name:
                content_cols.remove(cont)
                break
    else:
        content_cols = ['content']

    return content_cols


def parse_dataloader(data, args):
    torch_data = dict(
        img=torch.from_numpy(data['imgs']).permute(0, 3, 1, 2),
        img_id=torch.from_numpy(np.arange(data['imgs'].shape[0])),
        class_id=torch.from_numpy(data['classes'].astype(np.int64)),
        content=torch.from_numpy(data['contents'].astype(np.int64))
    )

    if landmarks_clause:
        torch_data['content'] = torch.from_numpy(data['landmarks'].astype(np.int64))

    workers = 4
    if not args.cuda:
        workers = 0

    dataset_ = NamedTensorDataset(torch_data)
    dataloader = DataLoader(
        dataset_, batch_size=32,
        shuffle=True, num_workers=workers,
        drop_last=False
    )

    return dataloader


# endregion DataSet Related Helpers

# region Model Related Helpers

def get_model(args, model_name, chosen_epoch):
    encoding_model = Encoder_Model(config)
    weights_prefix = 'encoder'
    model_dir = join(args.base_dir, 'models', model_name)
    weights_path = os.path.join(model_dir, f'{weights_prefix}_{chosen_epoch}.pth')
    if not os.path.exists(weights_path):
        return -1
    state_d = torch.load(weights_path)
    encoding_model.load_state_dict(state_d)
    encoding_model.to(device)
    encoding_model.eval()
    return encoding_model


def loop_once_gather_outputs(args, data_loader, model):
    class_ids = []
    indxs = []
    contents = []
    codes = []
    if args.eval_type == 'retrieval':
        imgs = []
    else:
        imgs = None

    pbar = tqdm(iterable=data_loader)
    for batch in pbar:

        batch = {name: tensor.to(args.device) for name, tensor in batch.items()}

        orig_out = model(batch['img'])
        # Note the regularization in case other models (besides DCoDR / DCoDR-norec) are added validated
        orig_out['code'] = F.normalize(orig_out['code'], dim=-1)
        codes.append(orig_out['code'].detach().cpu())
        indxs.append(batch['img_id'].detach().cpu())
        class_ids.append(batch['class_id'].detach().cpu())
        contents.append(batch['content'].detach().cpu())
        if args.eval_type == 'retrieval':
            imgs.append(batch['img'].detach().cpu())

    del batch
    del orig_out
    torch.cuda.empty_cache()
    gc.collect()

    return class_ids, indxs, contents, codes, imgs


def get_latent_codes(args, data_loader, model, content_cols):
    with torch.no_grad():
        class_ids, indxs, contents, codes, imgs = loop_once_gather_outputs(args, data_loader, model)

    class_ids = torch.cat(class_ids, dim=0).detach().cpu().numpy()
    indxs = torch.cat(indxs, dim=0).detach().cpu().numpy()
    contents = torch.cat(contents, dim=0).detach().cpu().numpy()
    if args.eval_type == 'retrieval':
        imgs = torch.cat(imgs, dim=0).detach().cpu().numpy().transpose(0, 2, 3, 1)

    contents = pd.DataFrame(contents, columns=content_cols, index=indxs)
    contents['class'] = class_ids  # Add class to contents
    contents = contents[['class'] + content_cols]  # set class to be first content

    codes = torch.cat(codes, dim=0)
    codes = pd.DataFrame(codes.detach().cpu().numpy(), index=indxs)

    torch.cuda.empty_cache()
    gc.collect()

    return class_ids, indxs, contents, codes, imgs


# endregion Model Related Helpers


# region Prediction Funcs

class Shallow_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=6):
        super(Shallow_Classifier, self).__init__()

        layers = []
        layers.extend(
            [
                nn.Linear(in_features=input_size,
                          out_features=hidden_size,
                          bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU()
            ]
        )
        for i in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(in_features=hidden_size,
                              out_features=hidden_size,
                              bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.LeakyReLU()
                ]
            )
        layers.append(nn.Linear(in_features=hidden_size,
                                out_features=output_size,
                                bias=True)
                      )
        self.predictor = nn.Sequential(*layers)
        return

    def forward(self, x):
        return self.predictor(x)

    def predict(self, x):
        acts = self.predictor(x)
        return F.softmax(acts, dim=1)


class DataFrame_Dataset(torch.utils.data.Dataset):

    def __init__(self, df: np.array, res: np.array, classes: np.array):
        self.vals = df.values
        self.res = res
        self.classes = classes

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i], self.res[i], self.classes[i]


def epoch_linear_classifier(model, dataloader, optimizer, is_classification=True):
    model.train()
    losses = []

    if is_classification:
        crit = nn.CrossEntropyLoss().to(device)
    else:
        crit = nn.L1Loss().to(device)
    for batch in dataloader:
        codes = batch[0].to(device)
        response = batch[1].to(device)  # .long()
        out = model(codes).double()
        loss = crit(out, response)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)


def classifier_prediction(model, dataloader, is_classification=True):
    model.eval()
    losses = []
    probs = []
    targets = []
    if is_classification:
        crit = nn.CrossEntropyLoss().to(device)
    else:
        crit = nn.L1Loss().to(device)
    for batch in dataloader:
        with torch.no_grad():
            codes = batch[0].to(device)
            response = batch[1].to(device)
            out = model(codes).double()
            if is_classification:
                probs.append(F.softmax(out, dim=-1))
            else:
                probs.append(out.detach().cpu())
            targets.append(response.detach().cpu())
            losses.append(crit(out, response).item())

    targets = torch.cat(targets).detach().cpu().numpy().astype(int)
    probs = torch.cat(probs, dim=0).detach().cpu().numpy()
    return np.mean(losses), probs, targets


def get_nn_classifier_essentials(inp_size, hidden_size, out_size, num_hidden_layers=6, device='cuda'):
    model = Shallow_Classifier(inp_size, hidden_size, out_size,
                               num_layers=num_hidden_layers).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.array([25, 40, 50, 60, 70, 80, 90]), 0.3)
    return model, optimizer, lr_scheduler


def predict_contents_nn(args, train_codes, train_contents,
                        val_codes, val_contents,
                        test_codes, test_contents, predicted_factors='all',
                        hidden_size=64, epochs=200, num_hidden_layers=6,
                        early_stopping=7):
    global weights_folder
    inp_size = train_codes.shape[1]
    dataset_constructor = DataFrame_Dataset

    if predicted_factors == 'all':
        if landmarks_clause:
            predicted_factors = ['class', 'landmarks']
        else:
            predicted_factors = list(train_contents.columns)

    all_accs = {}

    for factor_name in predicted_factors:

        if os.path.exists(f'{weights_folder}'):
            if args.delete_weights_folder:
                if os.path.exists(f'{weights_folder}'):
                    shutil.rmtree(f'{weights_folder}')
            else:
                counter = 0
                while os.path.exists(f'{weights_folder}_{counter}'):
                    counter += 1
                weights_folder = f'{weights_folder}_{counter}'
        os.makedirs(f'{weights_folder}', exist_ok=True)

        print(f'Starting training on {factor_name}')
        original_factor_name = deepcopy(factor_name)
        if landmarks_clause and factor_name == 'landmarks':
            is_classification = False
            factor_name = list(train_contents.drop('class', axis=1).columns)
            out_size = len(factor_name)
        else:
            is_classification = True
            out_size = len(np.unique(pd.concat([train_contents[factor_name],
                                                val_contents[factor_name],
                                                test_contents[factor_name]])
                                     )
                           )

        workers = 4
        if not args.cuda:
            workers = 0
        train_set = dataset_constructor(train_codes, train_contents[factor_name].values, train_contents['class'].values)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=False, num_workers=workers)
        val_set = dataset_constructor(val_codes, val_contents[factor_name].values, val_contents['class'].values)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=True, pin_memory=False, num_workers=workers)
        test_set = dataset_constructor(test_codes, test_contents[factor_name].values, test_contents['class'].values)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=False, num_workers=workers)
        if landmarks_clause and original_factor_name == 'landmarks':
            factor_name = 'landmarks'

        model, optimizer, lr_scheduler = get_nn_classifier_essentials(inp_size, hidden_size, out_size,
                                                                      num_hidden_layers=num_hidden_layers,
                                                                      device=args.device)

        train_losses, val_losses, val_accs = [], [], []
        pbar = tqdm(range(epochs))
        for epoch in pbar:

            train_ep_loss = epoch_linear_classifier(model, train_loader, optimizer, is_classification)
            lr_scheduler.step(train_ep_loss)
            torch.save(model.state_dict(), os.path.join(f'{weights_folder}', f'{epoch}.ckpt'))
            train_losses.append(train_ep_loss)

            ep_val_loss, val_probs, val_targets = classifier_prediction(model, val_loader, is_classification)
            val_losses.append(ep_val_loss)
            if is_classification:
                val_accs.append(np.mean(np.argmax(val_probs, axis=1) == val_targets))
                if epoch - np.argmax(val_accs) > early_stopping:
                    break
            elif epoch - np.argmin(val_losses) > early_stopping:
                break

            pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=ep_val_loss)

        optimizer.zero_grad()
        predictor_chosen_epoch = np.argmax(val_accs) if is_classification else np.argmin(val_losses)
        model.load_state_dict(torch.load(os.path.join(f'{weights_folder}', f'{predictor_chosen_epoch}.ckpt')))

        test_loss, test_probs, test_targets = classifier_prediction(model, test_loader, is_classification)
        print(f'Test Loss: {test_loss}')
        if is_classification:
            accuracy = np.mean(np.argmax(test_probs, axis=1) == test_targets)
            print(factor_name, '- prediction accuracy:', accuracy)
            all_accs[factor_name] = accuracy
        else:
            print(factor_name, '- prediction error:', test_loss)
            all_accs[factor_name + '_' + 'test_loss'] = test_loss

        print('\n\n\n')
    return all_accs


# endregion Prediction Funcs


if __name__ == '__main__':

    args = get_evaluation_args()
    landmarks_clause = 'celeba' in args.train_data_name

    print(f'evaluating experiments:\n')

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    args.device = device
    assets = AssetManager(args.base_dir)

    if 'celeba' in args.train_data_name:
        all_data, _, _ = load_np_data(assets, args.test_data_name, args.cuda)
        # remove small classes
        num_samples_by_class = pd.Series(all_data['classes']).value_counts()
        small_classes = list(num_samples_by_class[num_samples_by_class <= 1].index)
        keep_indxs = np.where(~pd.Series(all_data['classes']).isin(small_classes))[0]
        for k in all_data.keys():
            if k != 'n_classes':
                all_data[k] = all_data[k][keep_indxs]
        all_data['n_classes'] = len(np.unique(all_data['classes']))
        # make train and test by stratified split
        indxs = np.arange(all_data['imgs'].shape[0])
        cela_train_indxs, cela_test_indxs = train_test_split(indxs, stratify=all_data['classes'],
                                                             test_size=0.25, random_state=0)
        train_data, test_data = {}, {}
        for k in all_data.keys():
            if k == 'n_classes':
                train_data[k] = deepcopy(all_data[k])
                test_data[k] = deepcopy(all_data[k])
            else:
                train_data[k] = deepcopy(all_data[k][cela_train_indxs])
                test_data[k] = deepcopy(all_data[k][cela_test_indxs])
    else:
        train_data, _, _ = load_np_data(assets, args.train_data_name, args.cuda)
        test_data, _, _ = load_np_data(assets, args.test_data_name, args.cuda)
        all_data = {}
        for k in train_data.keys():
            if k == 'n_classes':
                all_data[k] = deepcopy(np.max([train_data[k], test_data[k]]))
            else:
                all_data[k] = deepcopy(np.concatenate([train_data[k], test_data[k]]))

    # DataLoaders

    trainloader = parse_dataloader(train_data, args)
    testloader = parse_dataloader(test_data, args)
    all_dataloader = parse_dataloader(all_data, args)

    content_cols = get_content_colnames(args)

    results_df = None

    for eval_counter, exp_name in enumerate(args.evaluated_exp_names):
        for chosen_epoch in args.chosen_epoch:
            # region Load Model

            print(f'\n\n\nEvaluating {exp_name} - {eval_counter}')

            exp_train_data = args.model_train_data if args.model_train_data is not None else args.train_data_name
            model_name = join(exp_train_data, exp_name)
            with open(join(assets.get_model_dir(model_name), 'config.pkl'), 'rb') as f:
                config = pickle.load(f)
            print(config)
            encoder_model = get_model(args, model_name, chosen_epoch)
            if encoder_model == -1:
                continue

            # endregion Load Model

            # region Encode All Examples

            print('Parsing Train Data...')
            trainval_class_ids, trainval_indxs, trainval_contents, trainval_codes, trainval_imgs = \
                get_latent_codes(args, trainloader, encoder_model, content_cols)
            print('Parsing Test Data...')
            test_class_ids, test_indxs, test_contents, test_codes, test_imgs = \
                get_latent_codes(args, testloader, encoder_model, content_cols)
            print('Parsing All Data...')
            all_class_ids, all_indxs, all_contents, all_codes, all_imgs = \
                get_latent_codes(args, all_dataloader, encoder_model, content_cols)

            if args.eval_type == 'prediction':
                del encoder_model
            torch.cuda.empty_cache()
            gc.collect()

            # endregion Encode All Examples

            if args.eval_type == 'prediction':

                # region Prepare split data

                # Split to train and val

                if 'celeba' not in args.train_data_name:
                    for factor_name in all_contents.columns:
                        mapping_dict = {x: i for i, x in enumerate(sorted(all_contents[factor_name].unique()))}
                        all_contents[factor_name] = all_contents[factor_name].apply(lambda x: mapping_dict[x])
                        trainval_contents[factor_name] = trainval_contents[factor_name].apply(lambda x: mapping_dict[x])
                        test_contents[factor_name] = test_contents[factor_name].apply(lambda x: mapping_dict[x])

                np.random.seed(2)
                train_indxs, val_indxs, train_cls, val_cls = train_test_split(trainval_indxs, trainval_class_ids,
                                                                              test_size=0.1)

                train_contents = trainval_contents.loc[train_indxs, :]
                val_contents = trainval_contents.loc[val_indxs, :]

                train_codes = trainval_codes.loc[train_indxs, :]
                val_codes = trainval_codes.loc[val_indxs, :]

                # endregion Prepare split data

                # Predict Contents

                if eval_counter == 0 and 'celeba' not in args.train_data_name:
                    for factor in all_contents.columns:
                        print(f'\n\nMajority: ({factor})')
                        num_unique = len(np.unique(all_contents[factor]))
                        val_counts = all_contents[factor].value_counts()
                        rolling_val = 0.
                        for j, val in enumerate(val_counts.index):
                            if j > 5:
                                break
                            rolling_val += val_counts[val]
                            print(f'\t{val}: {rolling_val / np.sum(val_counts)}')

                print('Test Prediction:')
                cur_results = predict_contents_nn(args, train_codes, train_contents,
                                                  val_codes, val_contents,
                                                  test_codes, test_contents,
                                                  epochs=100, hidden_size=128, num_hidden_layers=6,
                                                  predicted_factors='all'
                                                  )

                del train_codes, train_contents, val_codes, val_contents, test_codes, test_contents
            elif args.eval_type == 'retrieval':
                allowed_dists, modulo_sizes = get_retrieval_attrs_and_modulos(data_name=args.train_data_name,
                                                                              all_contents=all_contents,
                                                                              landmarks_clause=landmarks_clause)
                cur_results = get_retrieval_accuracies__from_all(args, all_contents, all_codes,
                                                                 test_contents, test_codes, test_class_ids,
                                                                 allowed_dists, modulo_sizes, dist=args.retrieval_dist)
                fig_dist = args.retrieval_dist if args.retrieval_dist != 'faiss' else 'l2'
                ret_fig = get_retrieval_figs(args, all_codes, all_class_ids, all_imgs,
                                             test_codes, test_class_ids, test_imgs,
                                             exp_name, chosen_epoch,
                                             k_neighbors=5, dist=fig_dist,
                                             num_samples=4, save_results_path=save_results_path)
            else:
                raise ValueError(f'No such eval type as {args.eval_type}')

            if results_df is None:
                results_df = pd.DataFrame(
                    -1 * np.ones((len(args.evaluated_exp_names), 1 + len(cur_results.keys())))).reset_index(drop=True)
                results_df.columns = ['exp_name'] + list(cur_results.keys())

            results_df.loc[eval_counter, 'exp_name'] = exp_name + f'__epoch_{chosen_epoch}'
            for k in cur_results.keys():
                results_df.loc[eval_counter, k] = cur_results[k]

    if os.path.exists(f'{weights_folder}'):
        # delete evaluation classifier weights
        shutil.rmtree(f'{weights_folder}')

    results_df.to_csv(
        join(save_results_path, f'{args.test_data_name}__{args.eval_name}__{args.eval_type}__results.csv'))
