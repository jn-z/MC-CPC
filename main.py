"""
main for LibriSpeech 
"""
## Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer
import pdb
## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

## Custrom Imports
from src.logger_v1 import setup_logs
from src.data_reader.dataset import RawDatasetSpkClass
from src.training_v1 import train, snapshot
from src.validation_v1 import validation
from src.model.model import CDCK2, CDCK5, CDCK3, CDCK6, CDCK_FUSION
############ Control Center and Hyperparameter ###############
run_name = "cpc"
print(run_name)

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-raw', required=True)
    parser.add_argument('--validation-raw', required=True)
    parser.add_argument('--eval-raw')
    parser.add_argument('--all-sets')
    parser.add_argument('--index-test-file')
    parser.add_argument('--index-file')
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--validation-list', required=True)
    parser.add_argument('--eval-list')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--frame-window', type=int, default=1)
    parser.add_argument('--audio-window', type=int, default=2100, 
                        help='window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12) 
    parser.add_argument('--masked-frames', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--CPCType',default='CDCK3',type=str)
    args = parser.parse_args()
    if (not os.path.exists(args.logging_dir)):
        os.makedirs(args.logging_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.CPCType == 'CDCK3':
       model = CDCK3(args.timestep, args.batch_size, args.audio_window).to(device)
    if args.CPCType == 'CDCK2':
       model = CDCK2(args.timestep, args.batch_size, args.audio_window).to(device)
    if args.CPCType == 'CDCK6':
       model = CDCK6(args.timestep, args.batch_size, args.audio_window).to(device)
    if args.CPCType == 'CDCK_FUSION':
       model = CDCK_FUSION(args.timestep, args.batch_size, args.audio_window).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}
    logger.info('===> loading train, validation and eval dataset')
    training_set   = RawDatasetSpkClass(args.train_raw, args.all_sets, args.train_list, args.index_file, args.audio_window, args.frame_window)
    validation_set = RawDatasetSpkClass(args.validation_raw, args.all_sets,args.validation_list, args.index_file, args.audio_window, args.frame_window)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
   # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_epoch = -1 
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)
        # Save
        if val_acc > best_acc: 
            best_acc = max(val_acc, best_acc)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
    ## end 
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
