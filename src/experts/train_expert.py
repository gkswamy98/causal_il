import gym
from stable_baselines3 import PPO

import sys
sys.path.append('..')
from src.lunar_lander_env import LunarLanderContinuous

if __name__ == "main":
    env = LunarLanderContinuous(confounding=False, fixed_terrain=False)
    model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=1e6)
    model.save("./ll_expert_2")