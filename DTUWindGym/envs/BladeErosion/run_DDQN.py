import sys
import json
from DDQN import DDQNAgent
from dtu_wind_gym.DTUWindGym.envs.BladeErosion.BladeErosionEnv import BladeErosionEnv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import gymnasium as gym
import xarray as xr
import pandas as pd
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import expon, truncnorm, uniform
from collections import deque
import time
import os
import pickle
import matplotlib.pyplot as plt


print(sys.argv)

for a in sys.argv:
    print(a)

update_every, gamma, tau, capacity, alpha, n_iter, save_freq = sys.argv


########### LOGGING #########
log_dir = "runs/run" + str(len(os.listdir("runs")))
os.makedirs(log_dir)
os.makedirs(log_dir + "/mdls")
# create new folder and log file for each run
log_f_name = log_dir + "/result_log.csv"
# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
# logging file
log_f = open(log_f_name, "w+")
log_f.write("episode,epsilon,reward,reward0,eval_mean,eval_std\n")

dictionary = {
    "start_time": str(start_time),
    "update_every": update_every,
    "discount_factor": gamma,
    "tau": tau,
    "capacity": capacity,
    "learning_rate": alpha,
    "n_iter": n_iter,
}
with open(log_dir + "/setting_log.json", "w") as outfile:
    json.dump(dictionary, outfile)


# Create the DDQN agent
action_space = np.linspace(0, 1, 21)
agent = DDQNAgent(
    state_size=3,
    action_size=len(action_space),
    seed=0,
    batch_size=512,
    update_every=update_every,
    discount_factor=gamma,
    tau=tau,
    capacity=capacity,
    learning_rate=alpha,
)
# Create env
env = BladeErosionEnv(fname="data/hornsrev1.nc")
# automatically compute power decay for exploration
eps_decay = -np.log(1e-3) / (n_iter * 0.95)
eps = 1
counter = 0
for i_episode in range(n_iter + 1):
    # save model
    if i_episode % save_freq == 0:
        torch.save(
            agent.qnetwork_local.state_dict(),
            log_dir + "/mdls/{}.pth".format(str(i_episode).zfill(6)),
        )
    # Initialize the environment and the state
    state = env.reset(percentile=np.random.uniform(low=1, high=99))
    score_no_damage = env.no_damage_reward()
    score = 0
    # Update the exploration rate
    if i_episode > (n_iter * 0.05):  # freeze for the first episodes
        counter += 1
        eps = np.exp(-(counter) * eps_decay)
    done = False
    # Run the episode
    while not done:
        # Select an action and take a step in the environment
        action_idx = agent.act(state, eps)
        action = action_space[action_idx]
        next_state, reward, done = env.step(action)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action_idx, reward, next_state, done)
        # Update the state and the score
        state = next_state
        score += reward
        # Break the loop if the episode is done or truncated
        if done:
            break
    # Save and print the rewards and scores
    print(
        "Episode {}, reward: {} ({:.5f}), epsilon: {:.3f}".format(
            i_episode, int(score), score / score_no_damage, eps
        )
    )
    log_f.write("{},{},{},{}\n".format(i_episode, eps, score, score_no_damage))
    log_f.flush()
# print total training time
end_time = datetime.now().replace(microsecond=0)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
log_f.close()

dictionary["end_time"] = str(end_time)
dictionary["duration"] = str(end_time - start_time)
with open(log_dir + "/setting_log.json", "w") as outfile:
    json.dump(dictionary, outfile)
