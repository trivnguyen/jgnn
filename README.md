# GraphNPE

**GraphNPE for inferring dark matter density profiles in dwarf galaxies**

This repository implements the GraphNPE method described in [Nguyen et al. (2025)](https://arxiv.org/abs/2503.03812), which uses graph neural networks combined with simulation-based inference to recover dark matter distributions from stellar kinematics in dwarf galaxies.

## Method

GraphNPE infers posterior distributions on dark matter density profiles directly from line-of-sight stellar velocities, without relying on traditional Jeans modeling assumptions. The method is trained on FIRE-2 simulations and works for both CDM and SIDM scenarios, achieving:

- Recovery of DM density profiles within 95% confidence with as few as ~30 tracers
- Peak circular velocity (V_peak) and virial mass (M_vir) estimation to 10-20% and 0.1-0.4 dex accuracy
- Robust performance even for tidally-disrupted systems

## Usage

The pipeline consists of three stages:

1. **Train embedding network**: `python train_embed.py --config=configs/embed.py`
2. **Train density estimator**: `python train_nde.py --config=configs/nde.py`
3. **Run inference**: Use the trained models to obtain posterior samples on DM parameters

See the paper for details on the methodology and validation.

## Citation

```bibtex
@article{nguyen2025trial,
  title={Trial by FIRE: Probing the dark matter density profile of dwarf galaxies with GraphNPE},
  author={Nguyen, Tri and Read, Justin and Necib, Lina and Mishra-Sharma, Siddharth and Faucher-Gigu{\`e}re, Claude-Andr{\'e} and Wetzel, Andrew and Starkenburg, Tjitske K},
  journal={arXiv preprint arXiv:2503.03812},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.