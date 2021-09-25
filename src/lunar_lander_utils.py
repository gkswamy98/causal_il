import numpy as np

def get_state(env):
    pos = env.lander.position
    vel = env.lander.linearVelocity
    state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (env.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            env.lander.angle,
            20.0*env.lander.angularVelocity/FPS,
    ]
    return np.array(state)

def set_state(env, s):
    env.lander.position.x = (s[0] * (VIEWPORT_W/SCALE/2)) + VIEWPORT_W/SCALE/2
    env.lander.position.y = (s[1] * (VIEWPORT_H/SCALE/2)) + (env.helipad_y+LEG_DOWN/SCALE)
    env.lander.linearVelocity.x = s[2] / ((VIEWPORT_W/SCALE/2)/FPS)
    env.lander.linearVelocity.y = s[3] / ((VIEWPORT_H/SCALE/2)/FPS)
    env.lander.angle = s[4] * 1.0
    env.lander.angularVelocity = s[5] * FPS / 20.0

def T(s, a, sim_env=sim_env):
    sim_env.reset()
    set_state(sim_env, s)
    obs, _, _, _ = sim_env.step(a)
    return obs

def dynamics(S, A, sim_env=sim_env):
    S_prime = []
    for (s, a) in zip(S, A):
        s_prime = T(s, a, sim_env)
        S_prime.append(s_prime)
    return np.stack(S_prime, axis=0)

def rollout(pi, env):
    states = []
    actions = []
    s = env.reset()
    done = False
    J = 0
    while not done:
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions, J