import numpy as np
import torch
from torch import optim
from copy import deepcopy

from .bc import BC

import sys
sys.path.append('..')
from src.models import Model

def DoubIL(D_E, pi_0, dynamics, lr=3e-4, nsamp=4, pi_BC=None):
    pi_init = deepcopy(pi_0)
    if pi_BC is None:
        pi_BC = BC(D_E, pi_0, lr=lr, steps=int(5e4))
    print("Done w/ BC")
    X_trajs = [x[0] for x in D_E]
    U_trajs = [x[1] for x in D_E]
    Z_IV = np.concatenate([xt[:-1] for xt in X_trajs], axis=0)
    U_IV = np.concatenate([ut[1:] for ut in U_trajs], axis=0)
    print('IV Data', Z_IV.shape, U_IV.shape)
    pi = pi_init
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(pi.parameters(), lr=lr)

    for step in range(int(5e4)):
        idx = np.random.choice(len(Z_IV), 128)
        actions = torch.from_numpy(U_IV[idx])
        optimizer.zero_grad()
        outputs_1 = 0
        outputs_2 = 0
        instruments = Z_IV[idx]
        bc_actions = pi_BC(torch.from_numpy(instruments).float()).detach().numpy()
        for i in range(int(nsamp / 2)):
            sample_actions = bc_actions + 3 * np.random.normal(size=(len(bc_actions), 1))
            states = torch.from_numpy(dynamics(instruments, sample_actions))
            with torch.no_grad():
                outputs_1 += pi(states.float())
        for i in range(int(nsamp / 2)):
            sample_actions = bc_actions + 3 * np.random.normal(size=(len(bc_actions), 1))
            states = torch.from_numpy(dynamics(instruments, sample_actions))
            outputs_2 += pi(states.float())
        outputs_1 = outputs_1 / (nsamp / 2)
        outputs_2 = outputs_2 / (nsamp / 2)
        factor_1 = (outputs_1 - actions.float()).detach()
        factor_2 = outputs_2 - actions.float()
        loss = torch.mean(factor_1 * factor_2)
        loss.backward()
        optimizer.step()
    return pi