import numpy as np
import torch
from torch import optim
from copy import deepcopy

from .bc import BC

import sys
sys.path.append('..')
from src.models import Model
import tqdm
import multiprocessing

def DoubIL(D_E, pi_0, dynamics, lr=3e-4, nsamp=4, pi_BC=None, wd=5e-3, sigma=6):
    pi_init = deepcopy(pi_0)
    if pi_BC is None:
        pi_BC = BC(D_E, pi_0, lr=lr)
    print("Done w/ BC")
    X_trajs = [x[0] for x in D_E]
    U_trajs = [x[1] for x in D_E]
    P = [x[2] for x in D_E]
    V = [x[3] for x in D_E]
    C = [x[4] for x in D_E]
    X_IV = []
    for i in range(nsamp):
        U_BC = [pi_BC(torch.from_numpy(xt[:-1]).float()).detach().numpy() + sigma * np.random.normal(size=(len(xt[:-1]), 1)) for xt in X_trajs]
        global samp
        def samp(j):
            return dynamics(P[j][:-1], V[j][:-1], C[j][:-1], U_BC[j][:-1], U_trajs[j][:-1])
        X_prime = []
        with multiprocessing.Pool() as pool:
            for result in pool.map(samp, list(range(len(D_E)))):
                X_prime.append(result)
        X_prime = np.concatenate(X_prime, axis=0)
        X_IV.append(X_prime) # samples from P(X|z)
    pi = pi_init
    U_IV = np.concatenate([ut[1:] for ut in U_trajs], axis=0) # single-sample estimate of E[Y|z]
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(pi.parameters(), lr=lr)
    print('IV Data', X_IV[0].shape, U_IV.shape)
    for step in tqdm.tqdm(range(int(5e4))):
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
