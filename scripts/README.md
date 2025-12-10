# Training Scripts

Canonical training and inference scripts for JGNN.

## Scripts

- `train_embed.py` - Train GNN embedding network
- `train_npe.py` - Train Neural Posterior Estimation
- `train_snpe.py` - Train Sequential NPE
- `train_optuna_npe.py` - Hyperparameter tuning with Optuna
- `test_snpe.py` - Test SNPE models
- `run_simulations.py` - Generate simulation datasets
- `plotting.py` - Visualization utilities

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
