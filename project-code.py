
import math
import random
import time

import gym
import pybulletgym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


from IPython.display import clear_output
import matplotlib.pyplot as plt


device   = torch.device("cpu")

#env = gym.make("Walker2DPyBulletEnv-v0")
#env_name = "InvertedPendulumPyBulletEnv-v0"
env_name = "Pendulum-v0"
env = gym.make(env_name)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
     

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            #nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

    
# Rename this fuck to evaluate agent
def test_env(vis):
    test_env_ = gym.make(env_name)
    #if vis: test_env_.render()
    state = test_env_.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = test_env_.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
        if vis: time.sleep(1/120)
        if vis: test_env_.render()
    
    #final_location_x = env.robot.get_location()[0]
    test_env_.close()
    #return total_reward, final_location_x
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def calculate_advantage(rewards, state_values, gamma, terminal_state):
    pass

"""
# Calculate the generalized advantage estimation
def calculate_gae(tau, time_horizon, gamma, advantages):
    gaes = []
    prev_gae = []

    # Start with the lastt sample and move backwards 
    for i in range(time_horizon, 0, -1):
        advantage = rewards[i] + gamma*state_values[i]*terminal_state[i] - next_state_values[i]
"""



def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20*16
mini_batch_size  = 20*16
ppo_epochs       = 4
threshold_reward = 100000
load_best_model   = False

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
if load_best_model:
    model.load_state_dict(torch.load('most_recent_model'))


max_frames = 1000000
frame_idx  = 0
best_model_score = -10**10


if load_best_model:
    with open('distance_travelled.npy', 'rb') as f:
        test_distances = list(np.load(f))
    with open('test_rewards.npy', 'rb') as f:
        test_rewards = list(np.load(f))
else:
    test_rewards = []
    test_distances = []


state = env.reset()
early_stop = False
done = False

while not early_stop:
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        if done:
            env.reset()

        state = state.reshape(1, -1)
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        next_state = state.reshape(1, -1)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        # try:
        #     reward = reward[0]
        # except e:
        #     pass
        #rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env(False) for _ in range(10)])
            test_rewards.append(test_reward)
            if test_reward > threshold_reward: early_stop = True
            print(test_reward)

        # if frame_idx % 1000 == 0:
        #     tst_rewards = []
        #     #distance_travelleds = []
        #     for i in range(10):
        #         #reward, distance_travelled = test_env(vis=False)
        #         reward = test_env(False)
        #         tst_rewards.append(reward)
        #         #distance_travelleds.append(distance_travelled)
        #     test_reward = np.mean(tst_rewards)
        #     #distance_travelled = np.mean(distance_travelleds)

        #     test_rewards.append(test_reward)
        #     #test_distances.append(distance_travelled)

        #     #print("score:", test_reward, "distance travelled:", distance_travelled)
        #     print("score:", test_reward)
        #     if test_reward > best_model_score: 
        #         best_model_score = test_reward
        #         torch.save(model.state_dict(), "best_model")
        #         with open('test_rewards.npy', 'wb') as f:
        #             np.save(f, test_rewards)
        #         #with open('distance_travelled.npy', 'wb') as f:
        #         #    np.save(f, test_distances)

        #     torch.save(model.state_dict(), "most_recent_model")

        #     if test_reward > threshold_reward: early_stop = True
            

    # Calculate the value of the next state
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
