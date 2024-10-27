import sys
sys.path.append("..")
import os
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataset import ICUDataset, SensorDataset, get_dataset
from utils.augmentations import Augmenter
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util_progress_log import AverageMeter, ProgressMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss

from models.models import ReverseLayerF

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import time
import shutil

import pickle
import json
import logging
import scipy.signal as signal

from argparse import ArgumentParser
from collections import namedtuple

from algorithms import get_algorithm

def main(args):
    # Usage example:
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    print('dataset shape:', dataset_src.sequence.shape)
    

    # For spectral_bandpass_filtering

    id = 0 # means the first sequence for source person
    # Assuming you want to save the frequency spectrum for the first sequence in the dataset
    fs = 20  # Sample rate (replace with your actual sample rate)

    for channel in range(3):
        save_path = f"Frequency_Spectrum_{id}_{channel}.jpg"
        dataset_src.save_frequency_spectrum(id, channel, fs, save_path)

    # For harmonic distrotion?


        

# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='cluda')

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99) #CLUDA
    parser.add_argument('-qs', '--queue_size', type=int, default=98304) #CLUDA
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true') #CLUDA
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--n_layers', type=int, default=1) #VRADA
    parser.add_argument('--h_dim', type=int, default=64) #VRADA
    parser.add_argument('--z_dim', type=int, default=64) #VRADA
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64') #All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3) #All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2) #All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1) #All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256) #All classifier and discriminators

    #The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=1.0)
    parser.add_argument('-w_kld', '--weight_KLD', type=float, default=1.0) #VRADA
    parser.add_argument('-w_nll', '--weight_NLL', type=float, default=1.0) #VRADA
    parser.add_argument('-w_cent', '--weight_cent_target', type=float, default=1.0) #CDAN
    #Below weights are defined for CLUDA
    parser.add_argument('--weight_loss_src', type=float, default=1.0)
    parser.add_argument('--weight_loss_trg', type=float, default=1.0)
    parser.add_argument('--weight_loss_ts', type=float, default=1.0)
    parser.add_argument('--weight_loss_disc', type=float, default=1.0)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')

    parser.add_argument('--path_src', type=str, default='../Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='../Data/aumc_fullstays')
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=int, default=1)
    parser.add_argument('--id_trg', type=int, default=2)

    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)