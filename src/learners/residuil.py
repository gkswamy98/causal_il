import numpy as np
import torch
from torch import optim

import sys
sys.path.append('..')
from src.models import Model
from src.oadam import OAdam, add_weight_decay, net_to_list
import tqdm

def ResiduIL(D_E, pi_0, f_0, lr=5e-5, f_norm_penalty=1e-3, bc_reg=5e-2, wd=1e-3):
    pi = pi_0
    f = f_0

    if isinstance(pi, Model):
        optimizer_pi = OAdam(add_weight_decay(pi, wd),
                                        lr=lr, betas=(0, .01))
        optimizer_f = OAdam(add_weight_decay(f, wd),
                                        lr=lr, betas=(0, .01))
    else:
        optimizer_pi = OAdam(net_to_list(pi),
                                        lr=lr, betas=(0, .01))
        optimizer_f = OAdam(net_to_list(f),
                                        lr=lr, betas=(0, .01))

    X_trajs = [x[0] for x in D_E]
    U_trajs = [x[1] for x in D_E]
    
    X = np.concatenate([xt[1:] for xt in X_trajs], axis=0)
    U = np.concatenate([ut[1:] for ut in U_trajs], axis=0)
    X_past = np.concatenate([xt[:-1] for xt in X_trajs], axis=0)

    for step in tqdm.tqdm(range(int(5e4))):
        idx = np.random.choice(len(X), 128)
        pi_inputs = torch.from_numpy(X[idx])
        f_inputs = torch.from_numpy(X_past[idx])
        targets = torch.from_numpy(U[idx]).float()

        optimizer_pi.zero_grad()
        preds = pi(pi_inputs.float())
        pred_residuals = f(f_inputs.float())
        loss = torch.mean(2 * (targets - preds) * pred_residuals)
        loss = loss + bc_reg * torch.mean(torch.square(targets - preds))
        loss.backward()
        optimizer_pi.step()
    
        optimizer_f.zero_grad()
        preds = pi(pi_inputs.float())
        pred_residuals = f(f_inputs.float())
        loss = -torch.mean(2 * (targets - preds) * pred_residuals - pred_residuals * pred_residuals)
        loss = loss + f_norm_penalty * torch.linalg.norm(pred_residuals)
        loss.backward()
        optimizer_f.step()
        
    return pi
