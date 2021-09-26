import numpy as np
import torch
from torch import optim, nn

import sys
sys.path.append('..')
from src.models import Model

def BC(D_E, pi_0, loss_fn=nn.MSELoss(), lr=3e-4, steps=int(5e4)):
    pi = pi_0
    X = np.concatenate([x[0] for x in D_E], axis=0)
    U = np.concatenate([x[1] for x in D_E], axis=0)
    print('BC Data', X.shape, U.shape)
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(pi.parameters(), lr=lr)

    for step in range(steps):
        idx = np.random.choice(len(X), 128)
        states = torch.from_numpy(X[idx])
        actions = torch.from_numpy(U[idx])

        optimizer.zero_grad()

        outputs = pi(states.float())
        loss = loss_fn(outputs, actions.float())
        loss.backward()
        optimizer.step()
    return pi