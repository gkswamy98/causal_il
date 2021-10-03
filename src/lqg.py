import numpy as np
from numpy.linalg import inv

dt = 0.1 # timestep
A = np.array([[1., dt], [0, 1.]])
B = np.array([[0.5 * dt * dt], [dt]])
Q = np.array([[1., 0], [0, 1.]])
R = np.array([[0.1]])

def dynamics(s, a):
    if len(s) == len(A): # single state
        return A @ s + B @ a
    else: # batch case
        return (s @ A.T + a @ B.T)

def solve(A, B, Q, R):
    P = np.zeros_like(Q)
    delta = 1.
    t = 0
    while delta > 0.01:
        K = - inv(R + B.T @ P @ B) @ B.T @ P @ A
        P_new = Q + (K.T @ R @ K) + (A + B @ K).T @ P @ (A + B @ K)
        delta = np.linalg.norm(P - P_new)
        P = P_new
        t += 1
    print('Converged with horizon', t)
    return K

def noise():
    return np.random.normal()

def rollout(s_0, pi, T=39):
    states = []
    actions = []
    s = s_0
    J = 0
    for _ in range(T):
        states.append(s.reshape(-1))
        a = pi(s)
        J += (s.T @ Q @ s) + (a.T @ R @ a)
        actions.append(a.reshape(-1))
        s = dynamics(s, a)
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions, J

def noisy_rollout(s_0, pi, T=100, num_steps_cnfnd=1):
    states = []
    actions = []
    s = s_0
    u_past = [0] * num_steps_cnfnd
    J = 0
    for t in range(T):
        states.append(s.reshape(-1))
        u = noise()
        a = pi(s) + (1 * u + 1 * np.sum(u_past)) / 1
        J += (s.T @ Q @ s) + (a.T @ R @ a)
        u_past[t % num_steps_cnfnd] = u
        actions.append(a.reshape(-1))
        s = dynamics(s, a)
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions, J