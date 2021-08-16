# Multi-task-Physics-informed-neural-networks

Research code base. The concept of Multi-task Physics-informed Neural Networks is first proposed in https://arxiv.org/abs/2104.14320.

The current directory corresponds to the Burgers' PDE discovery

See /inverse_KdV for the Korteweg-de Vries (KdV) PDE discovery

See /inverse_KS for the Kuramoto Sivashinsky (KS) PDE discovery

See /inverse_NLS for the Non-linear Schrodinger (NLS) (complex-valued) PDE discovery

See /inverse_qho for the Quantum Harmonic Oscillator (complex-valued) PDE discovery

For a small (lower data samples) version of KdV and KS, check out /inverse_small_KdV and /inverse_small_KS

This repo also additionally include the ladder networks (https://arxiv.org/abs/1507.02672) simple implementation in pytorch. See ladder.py

VAE & AutoEncoder implementation -> /vae_experiments

if you find this repo useful, please consider citing this!

## Citation
```
@article{thanasutives2021adversarial,
      title={Adversarial Multi-task Learning Enhanced Physics-informed Neural Networks for Solving Partial Differential Equations}, 
      author={Pongpisit Thanasutives and Masayuki Numao and Ken-ichi Fukui},
      year={2021},
      eprint={2104.14320},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
