import gym
from stable_baselines3 import PPO, SAC

import sys
sys.path.append('..')
from src.lunar_lander_env import LunarLanderContinuous


if __name__ == "main":
  parser = argparse.ArgumentParser(description='Train expert policies.')
  parser.add_argument('env', choices=['lunarlander', 'halfcheetah', 'ant'])
  args = parser.parse_args()
  
  if args.env == 'lunarlander':
    env = LunarLanderContinuous(confounding=False, fixed_terrain=False, sigma=0)
    model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]))
    for j in range(6):
        if j > 0:
            model.learn(total_timesteps=1e5)
        env.sigma += 0.1
    model.save("./ll_expert_curr")
  if args.env == 'halfcheetah':
    env = gym.make("HalfCheetahBulletEnv-v0")
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("./halfcheetah_expert")
  if args.env == 'ant':
    env = gym.make("AntBulletEnv-v0")
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("./ant_expert")

