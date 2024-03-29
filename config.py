import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


optimizer_dict = {
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}
activation_dict = {
    'elu': nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "rrelu": nn.RReLU,
    "tanh": nn.Tanh
}


class Config(object):
    def __init__(self, **kwargs):
        """ Configuration Class: set kwargs as class attributes with setattr """
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

    def __str__(self):
        """ Pretty-print configurations in alphabetical order """
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, default='mosei')
    parser.add_argument('--dataset_name', type=str, default='split_dataset_2')
    # parser.add_argument('--dataset_name', type=str, default='split_dataset_7classes_1')
    """
    mosei -> split_dataset_2 (mosei 2 classes)
    mosei -> split_dataset_7classes_1 (mosei 7 classes)

    mosi -> split_dataset_2 (mosi 2 classes)
    mosi -> split_dataset_7classes_1 (mosi 7 classes)
    """

    # Main config
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--model_index', type=int, default=233)
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--base_model', type=str, default='magbert_model')
    # parser.add_argument('--base_model', type=str, default='misa_model')
    # parser.add_argument('--base_model', type=str, default='selfmm_model')
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--model_savepath', type=str, default='./checkpoints')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--alpha', type=list, default=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0]) # 3, 2.75, 2.5, 2.25,
    parser.add_argument('--output_size', type=int, default=2)
    # parser.add_argument('--output_size', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--only_base_model', type=str2bool, default=False)
    parser.add_argument('--use_intervention', type=str2bool, default=True)
    parser.add_argument('--use_bert', type=str2bool, default=True)  # determine the poisonous words embedding
    parser.add_argument('--m_heads', type=int, default=8)

    # parser.add_argument('--fusion_mode', type=str, default='sum')
    parser.add_argument('--fusion_mode', type=str, default='hm')
    # parser.add_argument('--fusion_mode', type=str, default='rubi')
    # parser.add_argument('--fusion_mode', type=str, default='fc')
    parser.add_argument('--base_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--variance', type=float, default=0.1)
    parser.add_argument('--use_bert_frozen', type=str2bool, default=False)

    # TMODEL config
    parser.add_argument('--tmodel_hidden_size', type=int, default=128)
    parser.add_argument('--tmodel_embedding_size', type=int, default=32)
    parser.add_argument('--tmodel_rnncell', type=str, default='lstm')
    parser.add_argument('--tmodel_learning_rate', type=float, default=1e-5)  # origin
    parser.add_argument('--tmodel_weight_decay', type=float, default=1e-5)

    # MISA config
    parser.add_argument('--misa_rnncell', type=str, default='lstm')
    parser.add_argument('--misa_activation', type=str, default='relu')
    parser.add_argument('--misa_embedding_size', type=int, default=300)
    parser.add_argument('--misa_use_bert', type=str2bool, default=True)
    parser.add_argument('--misa_diff_weight', type=float, default=0.3)
    parser.add_argument('--misa_recon_weight', type=float, default=1.0)
    parser.add_argument('--misa_hidden_size', type=int, default=128)
    parser.add_argument('--misa_dropout', type=float, default=0.5)  # origin
    parser.add_argument('--misa_use_cmd_sim', type=str2bool, default=True)
    parser.add_argument('--misa_sim_weight', type=float, default=1.0)  # origin - 1.0
    parser.add_argument('--misa_sp_weight', type=float, default=0.0)  # origin - 0.0
    parser.add_argument('--misa_reverse_grad_weight', type=float, default=1.0)  # origin - 1.0
    parser.add_argument('--misa_learning_rate', type=float, default=1e-5)  # origin
    parser.add_argument('--misa_weight_decay', type=float, default=1e-5)

    # SELF-MM config
    parser.add_argument('--selfmm_train_samples', type=int, default=0)
    parser.add_argument('--selfmm_excludeZero', type=str2bool, default=True)
    parser.add_argument('--selfmm_H', type=float, default=3.0)
    parser.add_argument('--selfmm_video_hidden_size', type=int, default=32)
    parser.add_argument('--selfmm_audio_hidden_size', type=int, default=16)
    parser.add_argument('--selfmm_text_hidden_size', type=int, default=64)
    parser.add_argument('--selfmm_fusion_hidden_size', type=int, default=128)
    parser.add_argument('--selfmm_learning_rate', type=float, default=1e-5)  # origin
    parser.add_argument('--selfmm_weight_decay', type=float, default=1e-5)

    # MAG-BERT config
    parser.add_argument('--magbert_beta_shift', type=float, default=1)
    parser.add_argument('--magbert_dropout', type=float, default=0.3)  # origin
    parser.add_argument('--magbert_learning_rate', type=float, default=1e-5)  # origin
    parser.add_argument('--magbert_weight_decay', type=float, default=0)

    # extra, no need to change
    parser.add_argument('--split_rate', type=float, default=1)
    parser.add_argument('--min_number', type=int, default=300)
    parser.add_argument('--random_rate', type=float, default=1)
    parser.add_argument('--soft_label', type=float, default=False)
    parser.add_argument('--patience', type=int, default=10)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print('data = ', kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = kwargs.output_size
        kwargs.batch_size = 16  # 16
        kwargs.misa_reverse_grad_weight = 0.8
        kwargs.misa_diff_weight = 0.1
        kwargs.misa_sim_weight = 0.3
        kwargs.misa_sp_weight = 1.0
        kwargs.clip = 0.8
        kwargs.text_size = 768  # 768 / 300
        kwargs.audio_size = 74
        kwargs.video_size = 47

    elif kwargs.data == "mosei":
        kwargs.num_classes = kwargs.output_size
        kwargs.batch_size = 12  # 12
        kwargs.text_size = 768  # 768 / 300
        kwargs.audio_size = 74
        kwargs.video_size = 35
    else:
        print("No dataset mentioned")
        exit()

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
