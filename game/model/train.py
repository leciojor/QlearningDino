import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Qnet import QNetwork

import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
import torch.distributions as D
from Environment import Env


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_action_dqn(network, state, epsilon, epsilon_decay, action_size):

  state = torch.tensor(state, dtype=torch.float32).to(device)
  possible_a = [torch.argmax(network(state.unsqueeze(0)), dim=1).item(), random.choice(list(range(action_size)))]
  probs = torch.tensor([1-epsilon, epsilon])
  d = torch.distributions.Categorical(probs)
  action = possible_a[d.sample()]

  return action, epsilon*epsilon_decay


def prepare_batch(memory, batch_size, state_size):

  states = []
  actions = []
  next_states = []
  rewards = []
  dones = []
  for state, action, next_state, reward, done in memory:
    states.append(state)
    actions.append(action)
    next_states.append(next_state)
    rewards.append(reward)
    dones.append(done)

  return (
    torch.tensor(states, dtype=torch.float32),
    torch.tensor(actions, dtype=torch.long),
    torch.tensor(next_states, dtype=torch.float32),
    torch.tensor(rewards, dtype=torch.float32),
    torch.tensor(dones, dtype=torch.bool)
    )


def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update):


  state = batch[0].to(device)
  action = batch[1].to(device)
  next_state = batch[2].to(device)
  reward = batch[3].to(device)
  done = batch[4].to(device)
  batch_size = state[0].shape[0]

  with torch.no_grad():

    q = target_network(next_state).max(1)[0]
    y = reward + gamma * q * (1.0 - done.float())

  loss = torch.nn.functional.mse_loss(q_network(state).gather(1, action.unsqueeze(1)).squeeze(1), y)
  optim.zero_grad()
  loss.backward()
  optim.step()

  if not global_step % target_update:
    target_network.load_state_dict(q_network.state_dict())

def dqn_main(epochs, episode_limit, hidden_size, hidden_size_final, interval_save):
  lr = 1e-3
  start_training = 1000
  gamma = 0.99
  batch_size = 32
  epsilon = 1
  epsilon_decay = .9999
  target_update = 1000
  learn_frequency = 2

  state_size = 8
  action_size = 3
  env = Env()

  q_network = QNetwork(state_size, action_size, hidden_size, hidden_size_final).to(device)
  target_network = QNetwork(state_size, action_size, hidden_size, hidden_size_final).to(device)
  target_network.load_state_dict(q_network.state_dict())

  optim = torch.optim.Adam(q_network.parameters(), lr=lr)

  memory = []

  results_dqn = []
  global_step = 0
  loop = tqdm(total=epochs, position=0, leave=False)
  for epoch in range(epochs):
    last_epoch = (epoch+1 == epochs)
    # if last_epoch:
    #   env = wrap_env(env)

    state = env.reset()  
    done = False
    cum_reward = 0 

    while not done and cum_reward < episode_limit: 
      action, epsilon = get_action_dqn(q_network, state, epsilon, epsilon_decay, q_network.action_size)

      next_state, reward, terminated, _ = env.step(action) 
      done = terminated

      memory.append((state, action, next_state, reward, done))

      cum_reward += reward
      global_step += 1 
      state = next_state  

      if global_step > start_training and global_step % learn_frequency == 0:

        batch = prepare_batch(memory, batch_size, q_network.state_size)

        learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update)

    results_dqn.append(cum_reward)
    loop.update(1)
    loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))
    if not epoch % interval_save:
      torch.save(q_network.state_dict(), f"q_net_{epochs}_{episode_limit}_{hidden_size}_{hidden_size_final}_EPOCH:{epoch}.pkl") 
       
  torch.save(q_network.state_dict(), f"q_net_{epochs}_{episode_limit}_{hidden_size}_{hidden_size_final}.pkl")  
  return results_dqn
epochs = 3000
episode_limit = 800
hidden_size = 128
hidden_size_final = 64
results_dqn = dqn_main(epochs, episode_limit, hidden_size, hidden_size_final, 500)
plt.plot(results_dqn)
plt.savefig(f"q_net_{epochs}_{episode_limit}_plot.png")
plt.show()