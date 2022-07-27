# CausIL
A repo for Causal Imitation Learning under Temporally Correlated Noise.

## Running Experiments
To re-train an expert, run:
```bash
python experts/train_expert.py -e {lunarlander, halfcheetah, ant}
```
To train a learner, run:
```bash
jupyter notebook
```
and open up LunarLander.ipynb and PyBullet.ipynb. This package supports training via Behavioral Cloning, DoubIL, and ResiduIL.

## Visualizing Results
Run:
```bash
jupyter notebook
```
and open up vis.ipynb.

## Citing our Work
```
@article{swamy2021causal,
  author       = {Gokul Swamy and Sanjiban Choudhury and J. Andrew Bagnell and Zhiwei Steven Wu},
  title        = {Causal Imitation Learning under Temporally Correlated Noise},
  conference   = {Proceedings of the 39th International Conference on Machine Learning},
  url          = {https://arxiv.org/abs/2202.01312},
}
```
