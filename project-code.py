# This code is based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb

import math
import random

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


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv

num_envs = 16
env_name = "Walker2DPyBulletEnv-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

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
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            
            entropy_coefficient = 0.001
            distance,value = model(state)
            # Calculating the entropy for the yielded distance
            entropy = distance.entropy().mean()
            new_log_probs = distance.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            clipped_objective  = - torch.min(ratio*advantage, torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage)
            critic_loss = (return_ - value).pow(2)
            critic_mean_loss = critic_loss.mean()
            actor_mean_loss = clipped_objective.mean()
            loss = 0.5 * critic_mean_loss + actor_mean_loss - entropy_coefficient * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# This evaluates the agent
def evaluate_agent(vis, n_iterations):
    # vis: boolean, if the evaluation should be rendered
    # n_iterations: Integer, how many times the agent should be tested on the enviroment

    if vis: env.render()
    rewards = []
    distance_travelled = []

    for i in range(n_iterations):
        state = env.reset()
        in_terminal_state = False
        iteration_reward = 0
        while not in_terminal_state:
            state = torch.FloatTensor(state).reshape(1,-1).to(device)
            dist, _ = model(state)
            next_state, reward, in_terminal_state, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            iteration_reward += reward
            #time.sleep(1/60)
        
        distance_travelled.append(env.robot.get_location()[0])
        rewards.append(iteration_reward)

    

    return np.mean(rewards), np.mean(distance_travelled)


# Calculate the advantages
def calculate_advantage(rewards, state_values, gamma, terminal_state):
    # Rewards: A list of rewards
    # State_values: List of tensors of state_values
    # gamma: float
    # terminal_state: List of tensors of 0 or 1 if the corresponding state is a terminal state

    # Convert the lists into torch tensors
    next_values = torch.stack(state_values[1:])
    prev_values = torch.stack(state_values[0:-1])
    terminal_state = torch.stack(terminal_state)
    rewards = torch.stack(rewards)
    
    # Calculate the advantages
    advantage = rewards + gamma*next_values*terminal_state - prev_values
    return advantage



# Calculate the generalized advantage estimation
def calculate_gae(lambda_, num_steps, gamma, advantages, terminal_states):
    # lambda: float
    # num_steps: int, how many steps to calculate gae for
    # gamma: float
    # advantages: Tensor of advantages 
    # terminal_states: List of tensors of if the state is a terminal state

    gaes = []
    prev_gae = 0
    
    # Convert the lists to tensors
    terminal_states = torch.stack(terminal_states)

    # Start with the last sample and move backwards 
    for i in range(num_steps-1, -1, -1):
        gae = advantages[i] + gamma*lambda_*terminal_states[i]*prev_gae
        gaes.append(gae)
        prev_gae = gae
        
    # The list in the backwards order. Return it flipped
    return gaes[::-1]

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]


#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 256
mini_batch_size  = 64
ppo_epochs       = 10
discount_factor = 0.99
lambda_ = 0.95

time_steps_to_train_for = 10**6
evaluate_every = 10**3
test_rewards = []
best_model_score = -10**10
eval_visable = True
eval_n_iterations = 10

load_best_model   = False

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
if load_best_model:
    model.load_state_dict(torch.load('most_recent_model'))


if load_best_model:
    with open('distance_travelled.npy', 'rb') as f:
        test_distances = list(np.load(f))
    with open('test_rewards.npy', 'rb') as f:
        test_rewards = list(np.load(f))
else:
    test_rewards = []
    test_distances = []


state = envs.reset()
time_step_counter = 0



for time_step in range(time_steps_to_train_for//num_envs//num_steps):


    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    terminal_states     = []
    entropy = 0

    for step_counter in range(num_steps):
        time_step_counter+=1
        
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        terminal_states.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        
        if time_step_counter % evaluate_every == 0:
            test_reward, distance_travelled = evaluate_agent(eval_visable, eval_n_iterations) 
            test_rewards.append(test_reward)
            test_distances.append(distance_travelled)

            print("score:", test_reward, "distance travelled:", distance_travelled)
            if test_reward > best_model_score: 
                best_model_score = test_reward
                torch.save(model.state_dict(), "best_model")
                with open('test_rewards.npy', 'wb') as f:
                    np.save(f, test_rewards)
                with open('distance_travelled.npy', 'wb') as f:
                    np.save(f, test_distances)

            torch.save(model.state_dict(), "most_recent_model")
            

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    state_values = values + [next_value]
    advantages = calculate_advantage(rewards, state_values, discount_factor, terminal_states)
    gaes = calculate_gae(lambda_, num_steps, discount_factor, advantages, terminal_states)
    
    
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    gaes      = torch.cat(gaes).detach()
    returns   = (gaes + values).detach()
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, gaes)