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
