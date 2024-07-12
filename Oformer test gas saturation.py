#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:05:12 2024

@author: kint
"""

from sklearn.metrics import mean_absolute_error
from einops import rearrange
import matplotlib.pyplot as plt
from decoder_module import PointWiseDecoder2D
from encoder_module import SpatialTemporalEncoder2D
from lploss import *
from ufno import *
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
os.chdir('/home/kint/Music/ufno')


# %%
DATA_DIR = '/home/kint/Desktop/UFNO'

test_a = torch.load(f'{DATA_DIR}/sg_test_a.pt')
test_u = torch.load(f'{DATA_DIR}/sg_test_u.pt')

# %%


def build_model() -> (SpatialTemporalEncoder2D, PointWiseDecoder2D):
    # currently they are hard coded
    encoder = SpatialTemporalEncoder2D(
        input_channels=290,  # a + xy coordinates
        in_emb_dim=68,
        out_seq_emb_dim=84,
        heads=3,
        depth=5,
    )

    decoder = PointWiseDecoder2D(
        latent_channels=168,  # decoder_emb_dim
        out_channels=1,
        out_steps=8,
        propagator_depth=1,
        scale=8,
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
        sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


myloss = LpLoss(size_average=False)
# %%

# Create model instances
encoder, decoder = build_model()

# Load the model checkpoint - replace 'your_checkpoint_path' with the actual path
checkpoint_path = '/home/kint/Music/ufno/logs/model_ckpt/model_checkpoint_epoch_100.ckpt'
checkpoint = torch.load(checkpoint_path)

# Apply the state dictionaries from the checkpoint
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Move models to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

# %%

# Prepare for plotting
dx = np.cumsum(3.5938*np.power(1.035012, range(200))) + 0.1
X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))


def pcolor(x):
    plt.jet()
    return plt.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(x), shading='auto')


times = np.cumsum(np.power(1.421245, range(24)))
time_print = []
for t in range(times.shape[0]):
    if times[t] < 365:
        title = str(int(times[t]))+' d'
        time_print = np.append(time_print, title)
    else:
        title = f'{round(int(times[t])/365, 1)} y'
        time_print = np.append(time_print, title)


def dnorm_inj(a): return (a * (3e6 - 3e5) + 3e5) / (1e6 / 365*1000/1.862)
def dnorm_temp(a): return a * (180 - 30) + 30
def dnorm_P(a): return a * (300 - 100) + 100
def dnorm_lam(a): return a * 0.4 + 0.3
def dnorm_Swi(a): return a * 0.2 + 0.1


# %%
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a,
                                                                         test_u),
                                          batch_size=1,
                                          shuffle=False)

# %%


# Assuming that test_loader, device, encoder, and decoder are defined
total_r2 = 0
total_r2_plume = 0
total_mae = 0
total_mae_plume = 0
num_samples = 0
total_inference_time = 0

for i, (x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)
    x_plot = x.cpu().detach().numpy()
    y_plot = y.cpu().detach().numpy()

    # Apply padding and other preprocessing steps
    x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
              (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)

    # Mask calculation and other operations
    # Reservoir has different thickness as marked in the permeability map
    mask = x_plot[0, :, :, 0, 0] != 0  # mask.shape = (96,200)
    # sum(mask[0:,]) = 200. i.e. There is no padding in the x direction
    thickness = sum(mask[:, 0])

    batch_size = x.shape[0]
    size_x, size_y, size_t = test_a.shape[1], test_a.shape[2], test_a.shape[3]

    # Start timing before the neural network forward pass
    start_time = time.time()

    # Neural network forward pass and output processing
    grids = rearrange(x, 'b x y t c -> b (x y) t c')
    input_pos = prop_pos = grids[:, :, 0, -3:-1]  # [b (x y) 2]
    del grids

    time_channels = rearrange(
        x[:, :, :, :, 0:9], 'b x y t c -> b (x y) (t c)')  # [b (x y) (t c)]

    x = torch.cat((time_channels, input_pos), dim=-1)

    z = encoder.forward(x, input_pos)
    x_out = decoder.rollout(z, prop_pos, 32, input_pos)
    x_out = rearrange(x_out, 'b (t c) (h w) -> b h w t c',
                      h=size_x+8, w=size_y+8, t=size_t+8, c=1)
    x_out = x_out.view(batch_size, size_x+8, size_y+8,
                       size_t+8, 1)[..., :-8, :-8, :-8, :]
    x_out = x_out.squeeze(dim=-1)

    # Stop timing after the neural network forward pass
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    pred_plot = x_out.cpu().detach().numpy()

    # Flatten the arrays for R² and MRE calculation
    y_t = y_plot[0, :, :, :][mask]

    y_true = y_plot[0, :, :, :][mask].reshape(
        (thickness, -1))  # (thickness,4800), 4800 came from 24*200

    y_true_flat = y_true.flatten()

    y_p = pred_plot[0, :, :, :][mask]
    y_pred = pred_plot[0, :, :, :][mask].reshape((thickness, -1))

    y_pred_flat = y_pred.flatten()

    # Create mask where both original and predicted data are non-zero
    non_zero_mask = (y_true_flat != 0) & (y_pred_flat != 0)

    # Apply the mask to both arrays to filter out zeros
    filtered_y_true = y_true_flat[non_zero_mask]
    filtered_y_pred = y_pred_flat[non_zero_mask]

    # Calculate R² and MAE only on non-zero values
    # Ensure there are non-zero elements to avoid errors
    r2 = r2_score(y_true_flat, y_pred_flat)
    r2_plume = r2_score(filtered_y_true, filtered_y_pred)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mae_plume = mean_absolute_error(filtered_y_true, filtered_y_pred)
    print(f"R² score: {r2}")
    print(f"R² score (non-zero values only): {r2_plume}")
    print(f"MAE: {mae}")
    print(f"MAE (non-zero values only): {mae_plume}")

    # Accumulate the R² and MRE
    total_r2 += r2
    total_r2_plume += r2_plume
    total_mae += mae
    total_mae_plume += mae_plume
    num_samples += 1

    # Optionally break after a certain number of samples for testing
    if i >= test_a.shape[0]:
        break

# Calculate the average R² and MRE
average_r2 = total_r2 / num_samples
average_r2_plume = total_r2_plume / num_samples
average_mae = total_mae / num_samples
average_mae_plume = total_mae_plume / num_samples
average_inference_time = total_inference_time / num_samples

print(f'Average R²: {average_r2}')
print(f'Average R²_plume: {average_r2_plume}')
print(f'Average MAE: {average_mae}')
print(f'Average MAE_plume: {average_mae_plume}')
print(f'Average Inference Time: {average_inference_time} seconds')
# %%

# Assuming test_loader is already defined and loaded with all samples
for i, (x, y) in enumerate(test_loader):
    if i == 55:  # i-th samples, any number between 0 and 499
        x, y = x.to(device), y.to(device)
        x_plot = x.cpu().detach().numpy()
        y_plot = y.cpu().detach().numpy()

        # Apply padding and other preprocessing steps
        x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
                  (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)

        # Mask calculation and other operations
        mask = x_plot[0, :, :, 0, 0] != 0
        thickness = sum(mask[:, 0])

        batch_size = x.shape[0]
        size_x, size_y, size_t = test_a.shape[1], test_a.shape[2], test_a.shape[3]

        # Neural network forward pass and output processing
        grids = rearrange(x, 'b x y t c -> b (x y) t c')
        input_pos = prop_pos = grids[:, :, 0, -3:-1]  # [b (x y) 2]
        del grids

        time_channels = rearrange(
            x[:, :, :, :, 0:9], 'b x y t c -> b (x y) (t c)')  # [b (x y) (t c)]

        x = torch.cat((time_channels, input_pos), dim=-1)

        z = encoder.forward(x, input_pos)
        x_out = decoder.rollout(z, prop_pos, 32, input_pos)
        x_out = rearrange(x_out, 'b (t c) (h w) -> b h w t c',
                          h=size_x+8, w=size_y+8, t=size_t+8, c=1)
        x_out = x_out.view(batch_size, size_x+8, size_y+8,
                           size_t+8, 1)[..., :-8, :-8, :-8, :]
        x_out = x_out.squeeze(dim=-1)

        pred_plot = x_out.cpu().detach().numpy()

        # extract input parameters
        poro_map = x_plot[0, :, :, 0, 2][mask].reshape(
            (thickness, -1))  # shape = (thickness, 200)
        kr_map = np.exp(x_plot[0, :, :, 0, 0]
                        [mask].reshape((thickness, -1))*15)
        kz_map = np.exp(x_plot[0, :, :, 0, 1]
                        [mask].reshape((thickness, -1))*15)

        inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
        temperature = dnorm_temp(x_plot[0, 0, 0, 0, 5])
        pressure = dnorm_P(x_plot[0, 0, 0, 0, 6])
        Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
        lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])

        print(
            f'Params: injection rate: {inj_rate:.2f} MT/yr, temperature: {temperature:.1f} C, initial pressure: {pressure:.1f} bar, Swi: {Swi:.2f}, lan: {lam:.2f}')

        t_lst = [14, 20, 23]
        plt.figure(figsize=(15, 6))
        for j, t in enumerate(t_lst):
            plt.subplot(4, 3, j+1)
            if j == 2:
                pcolor(poro_map)
                plt.title('$\phi$ (-)')
            elif j == 1:
                pcolor(kz_map)
                plt.title('$k_z$ (mD)')
            else:
                pcolor(kr_map)
                plt.title('$k_r$ (mD)')
            plt.colorbar(fraction=0.02)
            plt.xlim([0, 3500])

            plt.subplot(4, 3, j+4)
            true_sg = y_plot[0, :, :, t][mask].reshape((thickness, -1))
            pcolor(true_sg)
            plt.title('$SG$ (-), '+f't={time_print[t]}')
            plt.colorbar(fraction=0.02)
            plt.xlim([0, 3500])

            # Plot predicted SG
            plt.subplot(4, 3, j+7)
            predicted_sg = pred_plot[0, :, :, t][mask].reshape((thickness, -1))
            pcolor(predicted_sg)
            plt.title('$\hat{SG}$ (-), '+f't={time_print[t]}')
            plt.colorbar(fraction=0.02)
            plt.xlim([0, 3500])
            print(predicted_sg.shape)

            # Prepare the true and predicted values for R^2 and MAE calculation
            true_sg_flat = y_plot[0, :, :, t][mask].reshape(
                (thickness, -1)).flatten()
            predicted_sg_flat = predicted_sg.flatten()

            # Create mask where both original and predicted data are non-zero
            non_zero_mask = (true_sg_flat != 0) & (predicted_sg_flat != 0)

            # Apply the mask to both arrays to filter out zeros
            filtered_y_true = true_sg_flat[non_zero_mask]
            filtered_y_pred = predicted_sg_flat[non_zero_mask]

            # Calculate R^2 and MAE
            r2 = r2_score(filtered_y_true, filtered_y_pred)
            mae = mean_absolute_error(filtered_y_true, filtered_y_pred)
            print(f"R² score (non-zero values only): {r2}")
            print(f"MAE (non-zero values only): {mae}")

            # Plot absolute error |dP - $\hat{dP}$|
            plt.subplot(4, 3, j+10)
            absolute_error = np.abs(predicted_sg - true_sg)
            pc = pcolor(absolute_error)
            plt.colorbar(pc, fraction=0.02)
            # Ensure the colorbar starts at 0
            plt.clim(0, np.max(absolute_error))
            plt.title('|$SG-\hat{SG}$|, ' + f't={time_print[t]}')
            plt.xlim([0, 3500])
        plt.tight_layout()
        plt.show()

# %%

# Assuming true_sg and predicted_sg are the saturation grids you're interested in
# This gives the index of the maximum value in flattened array
max_index_true = np.argmax(true_sg)
max_index_pred = np.argmax(predicted_sg)

# Convert flat indices to 2D indices
max_coords_true = np.unravel_index(max_index_true, true_sg.shape)
max_coords_pred = np.unravel_index(max_index_pred, predicted_sg.shape)

# Adjusting y-coordinate for the flipping
max_y_true_corrected = true_sg.shape[0] - 1 - max_coords_true[0]
max_y_pred_corrected = predicted_sg.shape[0] - 1 - max_coords_pred[0]

# Corrected coordinates
corrected_coords_true = (max_y_true_corrected, max_coords_true[1])
corrected_coords_pred = (max_y_pred_corrected, max_coords_pred[1])

print(f"Coordinates of max gas saturation in true_sg: {corrected_coords_true}")
print(
    f"Coordinates of max gas saturation in predicted_sg: {corrected_coords_pred}")
