import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader
from model_carrier import Model_Carrier

import torch
import torch.nn as nn
from torch.nn import functional as F

from create_dataset import Data_Reader, PAD, UNK

import optuna
from optuna.samplers import TPESampler, GridSampler


def return_unk():
    return UNK


def change(i):
    if i < -3:
        return 0
    elif i > 3:
        return 6
    else:
        return int(round(i)) + 3


def get_ban_list(data, train_config):
    ban_number = 0
    all_number = 0
    word_dict = {}

    for sample in data:
        label = sample[1][0][0]
        word_list = sample[0][3]
        all_number += len(word_list)

        for word in word_list:
            if train_config.output_size == 7:
                if word not in word_dict:
                    word_dict[word] = [0, 0, 0, 0, 0, 0, 0]
                new_label = change(label)
                word_dict[word][new_label] += 1

            if train_config.output_size == 2:
                if word not in word_dict:
                    word_dict[word] = [0, 0]
                if label > 0:
                    word_dict[word][0] += 1
                elif label < 0:
                    word_dict[word][1] += 1

    new_word_dict = {}

    if train_config.output_size == 2:
        for word, count_list in word_dict.items():
            if word not in new_word_dict:
                total = count_list[0] + count_list[1]
                if total <= train_config.min_number: continue
                new_word_dict[word] = abs(count_list[0] / total - count_list[1] / total)

    if train_config.output_size == 7:
        for word, count_list in word_dict.items():
            if word not in new_word_dict:
                total = 0
                for count in count_list:
                    total += count
                if total <= train_config.min_number: continue
                ban_number += total
                new_word_dict[word] = 0
                for count in count_list:
                    new_word_dict[word] += abs(count / total - 1.0 / 7.0)

    word_dict = sorted(new_word_dict.items(), key=lambda x: x[1], reverse=True)

    ban_word_list = []
    for word, rate in word_dict[int(len(word_dict) * train_config.split_rate):]:
        ban_word_list.append(word)

    ban_word_dict = {}
    for ban_word in ban_word_list:
        ban_word_dict[ban_word] = 0

    # print(all_number, ban_number)
    print('all_number = {}, ban_number = {}'.format(all_number, ban_number))
    return ban_word_dict


def init():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')


random_seed_list = [223, 123, 323]

if __name__ == '__main__':
    sampler = TPESampler(seed=1111)


    def objective(trial):
        train_config = get_config(mode='train')
        dev_config = get_config(mode='dev')
        test_iid_config = get_config(mode='test_iid')
        test_ood_config = get_config(mode='test_ood')

        dataset = Data_Reader(train_config)
        ban_word_list = []

        print('train_config = \n', train_config)

        iid_f1_nonzero_list = []
        iid_f1_list = []
        iid_acc2_nonzero_list = []
        iid_acc2_list = []
        iid_acc7_list = []
        f1_nonzero_list = []
        f1_list = []
        acc2_nonzero_list = []
        acc2_list = []
        acc7_list = []
        for random_seed in random_seed_list:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_seed)

            train_config.base_weight = trial.suggest_float('base_weight', 0.0, 1.0)
            train_config.ce_weight = trial.suggest_float('ce_weight', 0.0, 1.0)
            train_config.kl_weight = trial.suggest_float('kl_weight', 0.0, 1.0)

            train_loader, train_length = get_loader(train_config, shuffle=True, ban_word_list=ban_word_list)
            train_config.train_samples = train_length
            data_loader = {
                'train': train_loader,
                'dev': get_loader(dev_config, shuffle=False)[0],
                'test_iid': get_loader(test_iid_config, shuffle=False)[0],
                'test_ood': get_loader(test_ood_config, shuffle=False)[0],
            }
            model_carrier = Model_Carrier(train_config, data_loader, is_train=True)
            model_carrier.build()
            best_metrics = model_carrier.train()
            iid_f1_list.append(best_metrics['over']['iid']['f_score'])
            iid_f1_nonzero_list.append(best_metrics['over']['iid']['f_score_nonzero'])
            iid_acc2_list.append(best_metrics['over']['iid']['acc2'])
            iid_acc2_nonzero_list.append(best_metrics['over']['iid']['acc2_nonzero'])
            iid_acc7_list.append(best_metrics['over']['iid']['acc7'])
            f1_list.append(best_metrics['over']['ood']['f_score'])
            f1_nonzero_list.append(best_metrics['over']['ood']['f_score_nonzero'])
            acc2_list.append(best_metrics['over']['ood']['acc2'])
            acc2_nonzero_list.append(best_metrics['over']['ood']['acc2_nonzero'])
            acc7_list.append(best_metrics['over']['ood']['acc7'])
        print('=' * 50 + ' IID ' + '=' * 50)
        print("f1: ")
        print(f'avg: {np.mean(iid_f1_list)}, std: {np.std(iid_f1_list)}')
        print("f1_nonzero: ")
        print(f'avg: {np.mean(iid_f1_nonzero_list)}, std: {np.std(iid_f1_nonzero_list)}')
        print("acc2: ")
        print(f'avg: {np.mean(iid_acc2_list)}, std: {np.std(iid_acc2_list)}')
        print("acc2_nonzero: ")
        print(f'avg: {np.mean(iid_acc2_nonzero_list)}, std: {np.std(iid_acc2_nonzero_list)}')
        print("acc7: ")
        print(f'avg: {np.mean(iid_acc7_list)}, std: {np.std(iid_acc7_list)}')
        print('=' * 50 + ' OOD ' + '=' * 50)
        print("f1: ")
        print(f'avg: {np.mean(f1_list)}, std: {np.std(f1_list)}')
        print("f1_nonzero: ")
        print(f'avg: {np.mean(f1_nonzero_list)}, std: {np.std(f1_nonzero_list)}')
        print("acc2: ")
        print(f'avg: {np.mean(acc2_list)}, std: {np.std(acc2_list)}')
        print("acc2_nonzero: ")
        print(f'avg: {np.mean(acc2_nonzero_list)}, std: {np.std(acc2_nonzero_list)}')
        print("acc7: ")
        print(f'avg: {np.mean(acc7_list)}, std: {np.std(acc7_list)}')

        return (np.mean(acc2_list) + np.mean(acc2_nonzero_list)) / 2.0
        # return np.mean(acc7_list)

    n_trials = 100
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name='magbert-2', storage='sqlite:///ood.db', load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    print('-' * 50)
    print(f'Best Accuracy is: {best_trial.value:.4f}')
    print(f'Best Hyperparameters are:\n{best_trial.params}')
