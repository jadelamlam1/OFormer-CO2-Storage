#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 01:52:27 2024

@author: kint
"""

import gc
import logging
import datetime
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from einops import rearrange
import psutil
import optuna
from utils import load_checkpoint, save_checkpoint, ensure_dir
from decoder_module import PointWiseDecoder2D
from encoder_module import SpatialTemporalEncoder2D
from lploss import *
import os
os.chdir('/home/kint/Music/ufno')


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

def load_data(device):
    train_a = torch.load('/home/kint/Desktop/UFNO/sg_train_a.pt').float()
    train_u = torch.load('/home/kint/Desktop/UFNO/sg_train_u.pt').float()
    val_a = torch.load('/home/kint/Desktop/UFNO/sg_val_a.pt').float()
    val_u = torch.load('/home/kint/Desktop/UFNO/sg_val_u.pt').float()

    # Compute grid_x and grid_dx
    grid_x = train_a[0, 0, :, 0, -3].to(device)
    grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
    grid_dx = grid_dx[None, None, :, None].to(device)

    return train_a, train_u, val_a, val_u, grid_dx

# %%


def objective(trial, num_epochs, train_a, train_u, val_a, val_u, grid_dx, device):

    initial_batch_size = trial.suggest_categorical(
        'batch_size', [32, 16, 8, 4])
    batch_size = initial_batch_size

    while batch_size >= 4:
        try:
            # Model hyperparameters
            encoder_emb_dim = 4 * \
                trial.suggest_int('base_encoder_emb_dim', 15, 30)
            out_seq_emb_dim = 4 * \
                trial.suggest_int('base_out_seq_emb_dim', 20, 40)
            decoder_emb_dim = 2 * out_seq_emb_dim
            encoder_heads = trial.suggest_int('encoder_heads', 1, 6)
            encoder_depth = trial.suggest_int('encoder_depth', 2, 6)
            propagator_depth = trial.suggest_int('propagator_depth', 1, 3)
            scale = trial.suggest_int('fourier_frequency', 6, 10)

            # Training hyperparameters
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            aug_ratio = trial.suggest_float("aug_ratio", 0.1, 0.5)

            # Model setup
            opt = argparse.Namespace(
                in_channels=290,
                encoder_emb_dim=encoder_emb_dim,
                out_seq_emb_dim=out_seq_emb_dim,
                encoder_depth=encoder_depth,
                encoder_heads=encoder_heads,
                out_channels=1,
                decoder_emb_dim=decoder_emb_dim,
                out_step=32,
                propagator_depth=propagator_depth,
                fourier_frequency=scale,
                aug_ratio=aug_ratio,
                batch_size=batch_size,
                dataset_path='/home/kint/Desktop/UFNO',
                train_sample_num=4500,
                val_sample_num=500,
                lr=lr,
                ckpt_every=5,
                epochs=20  # Example: setting epochs to 10, adjust as needed
            )

            # Load data
            train_dataloader = DataLoader(TensorDataset(
                train_a, train_u), batch_size=opt.batch_size, shuffle=True)
            val_dataloader = DataLoader(TensorDataset(
                val_a, val_u), batch_size=opt.batch_size, shuffle=False)

            # Model instantiation
            encoder, decoder = build_model(opt)
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            encoder = encoder.to(device)
            decoder = decoder.to(device)

            # Define the total steps for scheduler
            total_steps = math.ceil(
                opt.train_sample_num / opt.batch_size) * opt.epochs
            print(f'Total steps:{total_steps}')

            # Initialize start_epoch
            start_epoch = 1

            # Create optimizers and schedulers
            enc_optim = torch.optim.AdamW(
                list(encoder.parameters()), lr=opt.lr, weight_decay=1e-4)
            dec_optim = torch.optim.AdamW(
                list(decoder.parameters()), lr=opt.lr, weight_decay=1e-4)

            enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=1e4, final_div_factor=1e4)
            dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=total_steps,
                                       div_factor=1e4, final_div_factor=1e4)

            # Initialize training times and validation loss list
            training_times = []
            epoch_train_losses = []
            epoch_val_losses = []

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

                    size_y, size_x, size_t = train_a.shape[1], train_a.shape[2], train_a.shape[3]

                    grids = rearrange(x, 'b y x t c -> b (y x) t c')

                    input_pos = prop_pos = grids[:, :, 0, -3:-1]  # [b (y x) 2]
                    del grids

                    # [4, (96*200), (24*12)]
                    input_channels = rearrange(
                        x[:, :, :, :, 0:9], 'b y x t c -> b (y x) (t c)')

                    if np.random.uniform() > (1-opt.aug_ratio):
                        sampling_ratio = np.random.uniform(0.45, 0.95)
                        input_idx = torch.as_tensor(np.concatenate([np.random.choice(input_pos.shape[1], int(sampling_ratio*input_pos.shape[1]), replace=False).reshape(
                            1, -1) for _ in range(input_channels.shape[0])], axis=0)).view(input_channels.shape[0], -1).cuda()

                        input_channels = index_points(
                            input_channels, input_idx)
                        input_pos = index_points(input_pos, input_idx)

                    x = torch.cat((input_channels, input_pos), dim=-1)

                    z = encoder.forward(x, input_pos)

                    x_out = decoder.rollout(z, prop_pos, size_t+8, input_pos)

                    x_out = rearrange(x_out, 'b (t c) (h w) -> b h w t c',
                                      h=size_y+8, w=size_x+8, t=size_t+8, c=1)

                    # Dynamically calculate the batch size
                    current_batch_size = x.shape[0]

                    x_out = x_out.view(current_batch_size, size_y+8, size_x+8,
                                       size_t+8, 1)[..., :-8, :-8, :-8, :]

                    x_out = x_out.squeeze()

                    ori_loss = 0
                    der_loss = 0

                    # original loss
                    for i in range(current_batch_size):
                        ori_loss += myloss(x_out[i, ...][mask[i, ...]].reshape(1, -1),
                                           y[i, ...][mask[i, ...]].reshape(1, -1))

                    # 1st derivative loss
                    dy_pred = (x_out[:, :, 2:, :] -
                               x_out[:, :, :-2, :])/grid_dx
                    mask_dy = mask[:, :, :198, :]
                    for i in range(current_batch_size):
                        der_loss += myloss(dy_pred[i, ...][mask_dy[i, ...]].reshape(
                            1, -1), dy[i, ...][mask_dy[i, ...]].view(1, -1))

                    loss = ori_loss + 0.5 * der_loss

                    enc_optim.zero_grad()
                    dec_optim.zero_grad()

                    loss.backward()
                    # log_cpu_memory_usage()  # Log memory usage after backward pass

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
                # log_cpu_memory_usage()

                # End timing the training phase
                end_train_time = time.time()
                train_duration = end_train_time - start_train_time

                # Store the training time for this epoch
                training_times.append(train_duration)

                # Print the average loss for the epoch
                print(
                    f'epoch: {epoch}, train loss: {train_l2/train_a.shape[0]:.4f}')

                # Validation phase
                # Every (default: 5) epochs, the model is assessed against 500 validation samples processed in batches of 4. This regular, systematic evaluation helps monitor and ensure the model's ability to generalize, guiding decisions about further training or adjustments to the model.
                if epoch % opt.ckpt_every == 0:
                    # logger.info('Validation')
                    print('Validation')

                    encoder.eval()
                    decoder.eval()

                    val_loss_epoch = []  # Store losses for the current validation epoch
                    val_ori_loss_epoch = []

                    with torch.no_grad():
                        for x, y in val_dataloader:
                            x, y = x.to(device), y.to(device)

                            dy = (y[:, :, 2:, :] - y[:, :, :-2, :])/grid_dx
                            mask = (x[:, :, :, 0:1, 0] !=
                                    0).repeat(1, 1, 1, 24)

                            x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
                                      (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)

                            size_y, size_x, size_t = val_a.shape[1], val_a.shape[2], val_a.shape[3]

                            grids = rearrange(
                                x, 'b y x t c -> b (y x) t c')  # [b (y x) 2]

                            input_pos = prop_pos = grids[:, :, 0, -3:-1]
                            del grids

                            input_channels = rearrange(
                                x[:, :, :, :, 0:9], 'b y x t c -> b (y x) (t c)')  # [b (y x) (t c)]

                            x = torch.cat((input_channels, input_pos), dim=-1)

                            z = encoder.forward(x, input_pos)
                            x_out = decoder.rollout(
                                z, prop_pos, size_t+8, input_pos)

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

            # Return the last validation loss
            last_val_loss = epoch_val_losses[-1]
            print(f'Last Validation Loss: {last_val_loss}')

            # After all epochs are complete, calculate the average training time per epoch
            average_training_time = sum(training_times) / len(training_times)
            print(
                f'Average training time per epoch: {average_training_time:.2f} seconds')

            # Reporting Peak Memory Usage
            peak_gpu_memory = torch.cuda.max_memory_allocated()/(1024 ** 3)
            print(
                f"Peak GPU memory usage during training: {peak_gpu_memory:.2f} GiB")

            # Return the last validation loss for optimization
            if epoch_val_losses:
                last_val_loss = epoch_val_losses[-1]
                print(f'Last Validation Loss: {last_val_loss}')
                return last_val_loss
            else:
                return float('inf')

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM Error: Reducing batch size from {batch_size}")
                batch_size //= 2  # Reduce batch size
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e  # For other types of errors, raise the exception

    print("Failed to process with all batch sizes due to OOM.")
    return float('inf')  # Return a large value after all retries fail

# %%


def run_coarse_tuning(trial, train_a, train_u, val_a, val_u, grid_dx, device):
    return objective(trial, num_epochs=20, train_a=train_a, train_u=train_u, val_a=val_a, val_u=val_u, grid_dx=grid_dx, device=device)


def run_fine_tuning(best_trial_params, train_a, train_u, val_a, val_u, grid_dx, device):
    study = optuna.create_study(direction='minimize')
    study.enqueue_trial(best_trial_params)

    def fine_tuning_objective(trial):
        return objective(trial, num_epochs=50, train_a=train_a, train_u=train_u, val_a=val_a, val_u=val_u, grid_dx=grid_dx, device=device)

    study.optimize(fine_tuning_objective, n_trials=50)

    print("Best trial after fine tuning:")
    print(study.best_trial.params)
    return study.best_trial

# %%


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_a, train_u, val_a, val_u, grid_dx = load_data(device)

    study = optuna.create_study(direction='minimize')

    try:
        # Pass all necessary data to run_coarse_tuning using a lambda function
        study.optimize(lambda trial: run_coarse_tuning(
            trial, train_a, train_u, val_a, val_u, grid_dx, device), n_trials=50)
    except KeyboardInterrupt:
        print("Optimization was stopped manually.")

    best_trial_params = study.best_trial.params
    print("Best trial so far:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in best_trial_params.items():
        print(f"{key}: {value}")

    # Now perform fine tuning
    best_trial_after_fine_tuning = run_fine_tuning(
        best_trial_params, train_a, train_u, val_a, val_u, grid_dx, device)
    print("Best parameters after fine tuning:")
    print(best_trial_after_fine_tuning.params)


if __name__ == '__main__':
    main()
