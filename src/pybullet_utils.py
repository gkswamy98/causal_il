import gym
import numpy as np
import tqdm

class CnfndWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, base_env, sigma=0.5):
        super(CnfndWrapper, self).__init__()
        self.base_env = base_env
        self.iter = 0
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.sigma = sigma
        self.u_past = np.random.normal() * self.sigma
    def step(self, action):
        u = np.random.normal() * self.sigma
        action = action + 1 * self.u_past + 1 * u
        self.u_past = u
        next_obs, reward, done, info = self.base_env.step(action)
        return next_obs, reward, done, info
    def reset(self):
        self.u_past = np.random.normal() * self.sigma
        obs = self.base_env.reset()
        return obs
    def render(self, mode='human'):
        self.base_env.render(mode=mode)
    def close (self):
        self.base_env.close()

def get_state(env):
    p = env.env._p
    base_pos = [] # position and orientation of base for each body
    base_vel = [] # velocity of base for each body
    joint_states = [] # joint states for each body
    for i in range(p.getNumBodies()):
        base_pos.append(p.getBasePositionAndOrientation(i))
        base_vel.append(p.getBaseVelocity(i))
        joint_states.append([p.getJointState(i,j) for j in range(p.getNumJoints(i))])
    return base_pos, base_vel, joint_states

def set_state(env, base_pos, base_vel, joint_states):
    p = env.env._p
    for i in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(i,*base_pos[i])
        p.resetBaseVelocity(i,*base_vel[i])
        for j in range(p.getNumJoints(i)):
            p.resetJointState(i,j,*joint_states[i][j][:2])

def T(base_pos, base_vel, joint_states, a, a_e, sim_env):
    sim_env.reset()
    set_state(sim_env, base_pos, base_vel, joint_states)
    for t in range(len(a_e)):
        sim_env.step(a_e[t])
    obs, _, _, _ = sim_env.step(a)
    return obs

def dynamics(P, V, C, A, A_exp, sim_env):
    S_prime = []
    for t in tqdm.tqdm(range(len(A))):
        a = A[t]
        a_e = A_exp[:t]
        s_prime = T(P[0], V[0], C[0], a, a_e, sim_env)
        S_prime.append(s_prime)
    return np.stack(S_prime, axis=0)

def rollout(pi, env, full_state=False):
    states = []
    actions = []
    if full_state:
        body_pos = []
        body_vel = []
        joint_states = []
    s = env.reset()
    if full_state:
        p, v, j = get_state(env)
        body_pos.append(p)
        body_vel.append(v)
        joint_states.append(j)
    done = False
    J = 0
    while not done:
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        if full_state:
            p, v, j = get_state(env)
            body_pos.append(p)
            body_vel.append(v)
            joint_states.append(j)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    if full_state:
        return states, actions, J, body_pos, body_vel, joint_states
    else:
        return states, actions, J

def noisy_rollout(pi, env, sigma=3, full_state=False):
    states = []
    actions = []
    if full_state:
        body_pos = []
        body_vel = []
        joint_states = []
    s = env.reset()
    if full_state:
        p, v, j = get_state(env)
        body_pos.append(p)
        body_vel.append(v)
        joint_states.append(j)
    done = False
    u_past = 0
    J = 0
    while not done:
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        u = np.random.normal() * sigma
        a = a + 1 * u + 1.0 * u_past
        u_past = u
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        if full_state:
            p, v, j = get_state(env)
            body_pos.append(p)
            body_vel.append(v)
            joint_states.append(j)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    if full_state:
        return states, actions, J, body_pos, body_vel, joint_states
    else:
        return states, actions, J
