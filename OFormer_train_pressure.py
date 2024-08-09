#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:27:52 2024

This script trains a OFormer model to predict the reservoir pressure after the injection of CO2.

Usage:
    python OFormer_train_pressure.py \
    --lr 1e-3 \
    # --resume_training \
    # --path_to_resume ./logs/model_ckpt \
    --epochs 140 \
    --log_dir ./logs \
    --ckpt_every 5 \
    --in_channels 290 \
    --encoder_emb_dim 68 \
    --out_seq_emb_dim 84 \
    --encoder_depth 5 \
    --encoder_heads 3 \
    --out_channels 1 \
    --decoder_emb_dim 168 \
    --out_step 32 \
    --propagator_depth 1 \
    --fourier_frequency 8 \
    --aug_ratio 0.1 \
    --batch_size 4 \
    --dataset_path /home/kint/Desktop/UFNO \
    --train_sample_num 4500 \
    --val_sample_num 500
    
    
Author:
    Jade
"""
# %%

# Standard Library Imports
import os
import logging
import datetime
import time
import math
import argparse

# Third-Party Library Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from einops import rearrange
import psutil

# Custom Module Imports
from utils import load_checkpoint, save_checkpoint, ensure_dir
from decoder_module import PointWiseDecoder2D
from encoder_module import SpatialTemporalEncoder2D
from lploss import *

# %%

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = device.type == 'cuda'
print(f'Using device: {device}')

# %%

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

# %%


def log_cpu_memory_usage():
    global peak_cpu_memory, peak_gpu_memory

    # Calculate current CPU and GPU memory usage in GiB
    total_memory = psutil.virtual_memory().total / (1024 ** 3)
    used_memory = (psutil.virtual_memory().total -
                   psutil.virtual_memory().available) / (1024 ** 3)

    # Update peak memory usage if current usage is higher
    peak_cpu_memory = max(peak_cpu_memory, used_memory)

# %%


# Print the PyTorch and CUDA versions
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")


def build_model(opt) -> (SpatialTemporalEncoder2D, PointWiseDecoder2D):
    encoder = SpatialTemporalEncoder2D(
        opt.in_channels,  # a + xy coordinates
        opt.encoder_emb_dim,
        opt.out_seq_emb_dim,  # Must be divisible by 4 to ensure Rotaryembedding works
        opt.encoder_heads,
        opt.encoder_depth,
    )

    decoder = PointWiseDecoder2D(
        opt.decoder_emb_dim,  # decoder_emb_dim, = double of out_seq_emb_dim
        opt.out_channels,
        opt.out_step,
        opt.propagator_depth,
        scale=opt.fourier_frequency,
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in encoder.parameters(
    ) if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


myloss = LpLoss(size_average=False)

# %%

# Create logs directory if it doesn't exist
log_dir_path = os.path.join(os.getcwd(), 'logs')
ensure_dir(log_dir_path)

# %%


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# %%


def get_arguments(parser):
    # basic training settings
    parser = argparse.ArgumentParser(description="Train a PDE transformer")

    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Specifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--resume_training', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str,
        default='none', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--epochs', type=int, default=140, help='Number of epochs. (default: 140)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./logs', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5, help='Save model checkpoints every x epochs. (default: 5)'
    )
    # ===================================
    # general option

    # model options for encoder
    parser.add_argument(
        '--in_channels', type=int, default=290, help='Channel of input feature. (default: 290)'
    )
    parser.add_argument(
        '--encoder_emb_dim', type=int, default=68, help='Channel of token embedding in encoder. (default: 68)'
    )
    parser.add_argument(
        '--out_seq_emb_dim', type=int, default=84, help='Channel of output feature map. (default: 84)'
    )
    parser.add_argument(
        '--encoder_depth', type=int, default=5, help='Depth of transformer in encoder. (default: 5)'
    )
    parser.add_argument(
        '--encoder_heads', type=int, default=3, help='Heads of transformer in encoder. (default: 3)'
    )
    # ===================================
    # model options for decoder
    parser.add_argument(
        '--out_channels', type=int, default=1, help='Channel of output. (default: 1)'
    )
    parser.add_argument(
        '--decoder_emb_dim', type=int, default=168, help='Channel of token embedding in decoder. (default: 168)'
    )
    parser.add_argument(
        '--out_step', type=int, default=1, help='How many steps to propagate forward each call. (default: 1)'
    )
    parser.add_argument(
        '--propagator_depth', type=int, default=1, help='Depth of mlp in propagator. (default: 1)'
    )
    parser.add_argument(
        '--fourier_frequency', type=int, default=8, help='Fourier feature frequency. (default: 8)'
    )
    parser.add_argument(
        '--aug_ratio', type=float, default=0.1, help='Probability to randomly crop. (default: 0.1)'
    )
    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Size of each batch (default: 4)'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='Path to dataset.'
    )
    parser.add_argument(
        '--train_sample_num', type=int, default=4500, help='How many samples in the training dataset.'
    )
    parser.add_argument(
        '--val_sample_num', type=int, default=500, help='How many samples in the validation dataset.'
    )
    return parser


# %%

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()

    print('Using following options')
    print(opt)

    # add code for datasets

    print('Preparing the data')

    # Load the training and validation datasets
    train_a = torch.load(f'{opt.dataset_path}/dP_train_a.pt')
    train_u = torch.load(f'{opt.dataset_path}/dP_train_u.pt')
    val_a = torch.load(f'{opt.dataset_path}/dP_val_a.pt')
    val_u = torch.load(f'{opt.dataset_path}/dP_val_u.pt')

    # Print the shapes of the loaded datasets
    print(f"train_a shape: {train_a.shape}")  # Print the shape of 'train_a'
    print(f"train_u shape: {train_u.shape}")  # Print the shape of 'train_u'
    print(f"val_a shape: {val_a.shape}")      # Print the shape of 'val_a'
    print(f"val_u shape: {val_u.shape}")      # Print the shape of 'val_u'

    grid_x = train_a[0, 0, :, 0, -3].to(device)
    grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
    grid_dx = grid_dx[None, None, :, None].to(device)

    # %%
    train_dataloader = DataLoader(TensorDataset(train_a, train_u),
                                  opt.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(TensorDataset(val_a, val_u),
                                opt.batch_size,
                                shuffle=False)
    # %%

    # instantiate network
    print('Building network')
    encoder, decoder = build_model(opt)

    # if running on GPU and we want to use cuda move model there
    if use_cuda:
        encoder, decoder = encoder.to(device), decoder.to(device)

    # saving checkpoints
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt')
    ensure_dir(checkpoint_dir)

    # save option information to the disk
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        '%s/%s.txt' % (opt.log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')

    # Define the total steps for scheduler
    total_steps = math.ceil(opt.train_sample_num / opt.batch_size) * opt.epochs
    print(f'Total steps:{total_steps}')

    # Initialize start_epoch
    start_epoch = 1

    # Create optimizers
    enc_optim = torch.optim.AdamW(
        list(encoder.parameters()), lr=opt.lr, weight_decay=1e-4)
    dec_optim = torch.optim.AdamW(
        list(decoder.parameters()), lr=opt.lr, weight_decay=1e-4)

    # Check if resuming from a checkpoint
    if opt.path_to_resume != 'none':
        print(f'Resuming checkpoint from: {opt.path_to_resume}')
        # Custom method for loading last checkpoint
        ckpt = load_checkpoint(opt.path_to_resume)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])

        if opt.resume_training:
            start_epoch = ckpt['epoch'] + 1
            # Calculate last epoch and steps
            # Use the stored last_epoch
            last_epoch = ckpt['enc_sched']['last_epoch']

            enc_optim.load_state_dict(ckpt['enc_optim'])
            dec_optim.load_state_dict(ckpt['dec_optim'])

            enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=1e4, final_div_factor=1e4, last_epoch=last_epoch)
            dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=1e4, final_div_factor=1e4, last_epoch=last_epoch)

            enc_scheduler.load_state_dict(ckpt['enc_sched'])
            dec_scheduler.load_state_dict(ckpt['dec_sched'])

            print("Pretrained checkpoint restored, training resumed")
            logger.info("Pretrained checkpoint restored, training resumed")

        else:
            enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=20, pct_start=0.05, final_div_factor=1e3)
            dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=20, pct_start=0.05, final_div_factor=1e3)

            print("Pretrained checkpoint restored, using tuning mode")
            logger.info("Pretrained checkpoint restored, using tuning mode")

    else:
        enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=total_steps,
                                   div_factor=1e4, final_div_factor=1e4)
        dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=total_steps,
                                   div_factor=1e4, final_div_factor=1e4)

        print("No pretrained checkpoint, starting training from scratch")
        logger.info("No pretrained checkpoint, starting training from scratch")

    # %%

    # Initialize training times and validation loss list
    training_times = []
    epoch_train_losses = []
    epoch_val_losses = []

    # Global variables for peak cpu memory usage
    peak_cpu_memory = 0

    # for loop going through dataset
    for epoch in range(start_epoch, opt.epochs+1):
        encoder.train()
        decoder.train()
        start_train_time = time.time()
        train_l2 = 0
        train_ori_l2 = 0
        counter = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            dy = (y[:, :, 2:, :] - y[:, :, :-2, :])/grid_dx
            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)

            x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
                      (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)

            size_x, size_y, size_t = train_a.shape[1], train_a.shape[2], train_a.shape[3]

            grids = rearrange(x, 'b x y t c -> b (x y) t c')
            input_pos = prop_pos = grids[:, :, 0, -3:-1]  # [b (x y) 2]
            del grids

            # [4, (96*200), (24*12)]
            input_channels = rearrange(
                x[:, :, :, :, 0:9], 'b x y t c -> b (x y) (t c)')

            if np.random.uniform() > (1-opt.aug_ratio):
                sampling_ratio = np.random.uniform(0.45, 0.95)
                input_idx = torch.as_tensor(np.concatenate([np.random.choice(input_pos.shape[1], int(sampling_ratio*input_pos.shape[1]), replace=False).reshape(
                    1, -1) for _ in range(input_channels.shape[0])], axis=0)).view(input_channels.shape[0], -1).cuda()

                input_channels = index_points(input_channels, input_idx)
                input_pos = index_points(input_pos, input_idx)

            x = torch.cat((input_channels, input_pos), dim=-1)

            z = encoder.forward(x, input_pos)

            x_out = decoder.rollout(z, prop_pos, size_t + 8, input_pos)

            x_out = rearrange(x_out, 'b (t c) (h w) -> b h w t c',
                              h=size_x+8, w=size_y+8, t=size_t+8, c=1)

            # Dynamically calculate the batch size
            current_batch_size = x.shape[0]

            x_out = x_out.view(current_batch_size, size_x+8, size_y+8,
                               size_t+8, 1)[..., :-8, :-8, :-8, :]
            x_out = x_out.squeeze()

            ori_loss = 0
            der_loss = 0

            # original loss
            for i in range(current_batch_size):
                ori_loss += myloss(x_out[i, ...][mask[i, ...]].reshape(1, -1),
                                   y[i, ...][mask[i, ...]].reshape(1, -1))

            # 1st derivative loss
            dy_pred = (x_out[:, :, 2:, :] - x_out[:, :, :-2, :])/grid_dx
            mask_dy = mask[:, :, :198, :]
            for i in range(current_batch_size):
                der_loss += myloss(dy_pred[i, ...][mask_dy[i, ...]].reshape(
                    1, -1), dy[i, ...][mask_dy[i, ...]].view(1, -1))

            loss = ori_loss + 0.5 * der_loss

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()
            log_cpu_memory_usage()  # Log memory usage after backward pass

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.)

            # Update weights and learning rate schedulers
            enc_optim.step()
            dec_optim.step()
            enc_scheduler.step()  # Scheduler step called per batch
            dec_scheduler.step()  # Scheduler step called per batch

            # Add loss to the cumulative sum for the epoch
            train_l2 += loss.item()
            train_ori_l2 += ori_loss.item()
            counter += 1

        # Calculate average losses for the epoch
        epoch_avg_train_loss = train_l2 / counter
        epoch_avg_train_ori_loss = train_ori_l2 / counter
        epoch_train_losses.append(epoch_avg_train_loss)

        # Log memory usage at the end of each epoch
        log_cpu_memory_usage()

        # End timing the training phase
        end_train_time = time.time()
        train_duration = end_train_time - start_train_time

        # Store the training time for this epoch
        training_times.append(train_duration)

        # Print the average loss for the epoch
        print(f'epoch: {epoch}, train loss: {train_l2/train_a.shape[0]:.4f}')

        # Validation phase
        # Every (default: 5) epochs, the model is assessed against 500 validation samples processed in batches of 4. This regular, systematic evaluation helps monitor and ensure the model's ability to generalize, guiding decisions about further training or adjustments to the model.
        if epoch % opt.ckpt_every == 0:
            logger.info('Validation')
            print('Validation')

            encoder.eval()
            decoder.eval()

            val_loss_epoch = []  # Store losses for the current validation epoch
            val_ori_loss_epoch = []

            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)

                    mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)
                    dy = (y[:, :, 2:, :] - y[:, :, :-2, :])/grid_dx

                    x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
                              (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)

                    grids = rearrange(
                        x, 'b x y t c -> b (x y) t c')  # [b (x y) 2]

                    input_pos = prop_pos = grids[:, :, 0, -3:-1]
                    del grids

                    input_channels = rearrange(
                        x[:, :, :, :, 0:9], 'b x y t c -> b (x y) (t c)')  # [b (x y) (t c)]

                    x = torch.cat((input_channels, input_pos), dim=-1)

                    size_y, size_x, size_t = val_a.shape[1], val_a.shape[2], val_a.shape[3]

                    z = encoder.forward(x, input_pos)
                    x_out = decoder.rollout(z, prop_pos, size_t + 8, input_pos)

                    x_out = x_out = rearrange(
                        x_out, 'b (t c) (h w) -> b h w t c', h=size_y+8, w=size_x+8, t=size_t+8, c=1)

                    # Dynamically calculate the batch size
                    current_batch_size = x.shape[0]

                    x_out = x_out.view(
                        current_batch_size, size_y+8, size_x+8, size_t+8, 1)[..., :-8, :-8, :-8, :]
                    x_out = x_out.squeeze()

                    # Compute losses as in training phase
                    ori_loss = 0
                    der_loss = 0

                    # Original loss
                    for i in range(current_batch_size):
                        ori_loss += myloss(x_out[i, ...][mask[i, ...]].reshape(
                            1, -1), y[i, ...][mask[i, ...]].reshape(1, -1))

                    # 1st derivative loss
                    dy_pred = (x_out[:, :, 2:, :] -
                               x_out[:, :, :-2, :])/grid_dx
                    mask_dy = mask[:, :, :198, :]
                    for i in range(current_batch_size):
                        der_loss += myloss(dy_pred[i, ...][mask_dy[i, ...]].reshape(
                            1, -1), dy[i, ...][mask_dy[i, ...]].view(1, -1))

                    loss = ori_loss + 0.5 * der_loss
                    val_loss_epoch.append(loss.item())
                    val_ori_loss_epoch.append(ori_loss.item())

            # Calculate average loss for this validation epoch
            val_loss_avg = np.mean(val_loss_epoch)
            val_ori_loss_avg = np.mean(val_ori_loss_epoch)

            # Append the average validation loss to the list for later plotting
            epoch_val_losses.append(val_loss_avg)

            print(f'Validation loss at epoch {epoch}: {val_loss_avg}')

            logger.info(f'Current epoch: {epoch}')
            logger.info(f'Validation loss at epoch {epoch}: {val_loss_avg}')

            # save checkpoint if needed
            ckpt = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'epoch': epoch,
                'enc_optim': enc_optim.state_dict(),
                'dec_optim': dec_optim.state_dict(),
                'enc_sched': enc_scheduler.state_dict(),
                'dec_sched': dec_scheduler.state_dict(),
            }

            save_checkpoint(ckpt, os.path.join(
                checkpoint_dir, f'model_checkpoint_epoch_{epoch}.ckpt'))
            del ckpt

    # After all epochs are complete, calculate the average training time per epoch
    average_training_time = sum(training_times) / len(training_times)
    print(
        f'Average training time per epoch: {average_training_time:.2f} seconds')

# Reporting Peak Memory Usage
peak_gpu_memory = torch.cuda.max_memory_allocated()/(1024 ** 3)
print(f"Peak CPU memory usage during training: {peak_cpu_memory:.2f} GiB")
print(f"Peak GPU memory usage during training: {peak_gpu_memory:.2f} GiB")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_train_losses, label='Training Loss',
         color='blue')  # removed marker='o'
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.grid(True)
plt.legend()
plt.show()

# Plot validation loss
plt.figure(figsize=(10, 5))

# Assuming validation starts at epoch 5 and repeats every 5 epochs
validation_epochs = np.arange(
    max(start_epoch-1, opt.ckpt_every), opt.epochs + 1, opt.ckpt_every)
plt.plot(validation_epochs, epoch_val_losses,
         label='Validation Loss', color='red')  # removed marker='o'
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
# Set x-ticks to show every epoch where validation was performed
plt.xticks(validation_epochs)
plt.grid(True)
plt.legend()
plt.show()
