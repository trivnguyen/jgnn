# Training Scripts

Canonical training and inference scripts for JGNN.

## Scripts

- `train_embed.py` - Train GNN/Transformer embedding network
- `train_npe.py` - Train Neural Posterior Estimation

Shared setup/callback/fit-loop logic lives in `jgnn.training` so both scripts
stay thin; each script only defines its own data prep and model construction.

## Usage

```bash
# Install jgnn in editable mode
pip install -e .

# Run with example config
python scripts/train_npe.py --config=configs/example_npe.py

# Run with your workspace config
python scripts/train_npe.py --config=workspace/configs/5params_npe/transformer_npe.py
```

## Customizing Scripts

If you need to modify a training script for your experiments:

```bash
# Copy to workspace
cp scripts/train_npe.py workspace/scripts/

# Edit and run from workspace
python workspace/scripts/train_npe.py --config=workspace/configs/my_config.py
```
