The above code is used to run the OFormer model for predicting CO2 gas saturation and pressure build-up. The dataset utilized for these predictions is sourced from [this repository](https://github.com/gegewen/ufno/tree/main). To proceed, download the following files: `utils.py`, `attention.py`, `encoder_module.py`, `decoder_module.py`, and `lploss.py`. Then, run either `OFormer_train_gas_saturation.py` or `OFormer_train_pressure.py`. Parameters can be adjusted in these two files for optimization.

## Testing Instructions

### Gas Saturation

Use the pre-trained model `model_checkpoint_epoch_100.py`:

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

    
### Pressure build-up

Use the pre-trained model `model_checkpoint_epoch_140.py`:

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
