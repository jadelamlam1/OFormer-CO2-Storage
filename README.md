# OFormer: Predicting CO2 Gas Saturation and Pressure Build-up

This project uses the OFormer model to predict CO2 gas saturation and pressure build-up.

## Requirements

- PyTorch 2.0.1
  
## Dataset

The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing), provided by Wen et al. in the [U-FNO GitHub repository](https://github.com/gegewen/ufno).

- **Train Set**: 4,500 samples
- **Validation Set**: 500 samples
- **Test Set**: 500 samples


## Training

Ensure you have the following files: 

- `utils.py`
- `attention.py`
- `encoder_module.py`
- `decoder_module.py`
- `lploss.py`

Then, run one of the following scripts to train the model:

- `OFormer_train_gas_saturation.py`
- `OFormer_train_pressure.py`

You can adjust parameters within these files for optimization.

## Testing Instructions

### Gas Saturation

Use the pre-trained model `model_checkpoint_epoch_100.ckpt`:

```bash
python OFormer_test_gas_saturation.py \
  --in_channels 290 \
  --encoder_emb_dim 68 \
  --out_seq_emb_dim 84 \
  --encoder_depth 5 \
  --encoder_heads 3 \
  --out_channels 1 \
  --decoder_emb_dim 168 \
  --out_step 1 \
  --propagator_depth 1 \
  --fourier_frequency 8 \
  --dataset_path /path/to/dataset \       # Change to your dataset path
  --checkpoint_path /path/to/checkpoints \     # Change to your checkpoint path
  --checkpoint_name model_checkpoint_epoch_100.ckpt
```

![CO2_movement](https://github.com/user-attachments/assets/0fadf42c-6926-4e49-aea8-1a92923c735b)


### Pressure Build-up

Use the pre-trained model `model_checkpoint_epoch_140.ckpt`:

```bash
python OFormer_test_pressure.py \
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
  --dataset_path /path/to/dataset \       # Change to your dataset path
  --checkpoint_path /path/to/checkpoints \    # Change to your checkpoint path
  --checkpoint_name model_checkpoint_epoch_140.ckpt
```

![dP_movement](https://github.com/user-attachments/assets/c656c334-d4d6-4a52-a623-b5ad4ffe52d1)

