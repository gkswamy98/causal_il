import numpy as np
import torch
from torch import optim
from copy import deepcopy

from .bc import BC

import sys
sys.path.append('..')
from src.models import Model

def DoubIL(D_E, pi_0, dynamics, lr=3e-4, nsamp=64, pi_BC=None):
    pi_init = deepcopy(pi_0)
    if pi_BC is None:
        pi_BC = BC(D_E, pi_0, lr=lr)
    print("Done w/ BC")
    X_trajs = [x[0] for x in D_E]
    U_trajs = [x[1] for x in D_E]
    X_IV = []
    U_IV = []
    for _ in range(nsamp):
        U_BC = [pi_BC(torch.from_numpy(xt[:-1]).float()).detach().numpy() + 3 * np.random.normal(size=(len(xt[:-1]), 1)) for xt in X_trajs]
        X_prime = np.concatenate([dynamics(X_trajs[i][:-1], U_BC[i]) for i in range(len(D_E))], axis=0)
        U_prime = pi_BC(torch.from_numpy(X_prime).float()).detach().numpy() + 3 * np.random.normal(size=(len(X_prime), 1))
        X_IV.append(X_prime) # samples from P(X|z)
        U_IV.append(U_prime)
    pi = pi_init
    if isinstance(pi, Model):
        U_IV = np.mean(U_IV, axis=0) # E[Y|Z]
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=1e-3)
    else:
        U_IV = np.concatenate([ut[1:] for ut in U_trajs], axis=0) # sufficient for linear case
        optimizer = optim.Adam(pi.parameters(), lr=lr)
    print('IV Data', X_IV[0].shape, U_IV.shape)
    for step in range(int(5e4)):
        idx = np.random.choice(len(X_IV[0]), 128)
        actions = torch.from_numpy(U_IV[idx])
        optimizer.zero_grad()
        outputs_1 = 0
        outputs_2 = 0
        sample_idx = list(range(nsamp))
        np.random.shuffle(sample_idx)
        for i in range(int(nsamp / 2)):
            states_1 = torch.from_numpy(X_IV[sample_idx[i]][idx])
            states_2 = torch.from_numpy(X_IV[sample_idx[i + int(nsamp / 2)]][idx])
            with torch.no_grad():
                outputs_1 += pi(states_1.float())
            outputs_2 += pi(states_2.float())
        outputs_1 = (outputs_1 / (nsamp / 2)).detach()
        outputs_2 = outputs_2 / (nsamp / 2)
        factor_1 = (outputs_1 - actions.float()).detach()
        factor_2 = outputs_2 - actions.float()
        loss = torch.mean(factor_1 * factor_2)
        loss.backward()
        optimizer.step()
    return pi