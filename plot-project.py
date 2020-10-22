# This code is heavily influenced by user Josh Albert's response on this stackoverflow question: https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho/47878011

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

test_rewards = []
distance_travlled = []
with open("distance_travelled-1-million.npy", 'rb') as f:
    distance_travlled = np.load(f)
with open("test_rewards-1-million.npy", 'rb') as f:
    test_rewards = np.load(f)


# Plot reward
plt.subplot(2,1,1)
mean = np.mean(rolling_window(test_rewards, 10), axis=-1)
var = np.var(rolling_window(test_rewards, 10), axis=-1)
std = np.sqrt(var)
x = np.arange(0.0, test_rewards.size, 1)

plt.plot(mean, color='b')
plt.fill_between(x, mean+std, mean-std, alpha=0.3, color='b')

reward_patch_moving_avg_mu = mpatches.Patch(color='b', label='Rolling average reward')
reward_patch_moving_avg_std = mpatches.Patch(color='b', alpha=0.5, label='Rolling average reward std')
plt.legend(handles=[reward_patch_moving_avg_mu, reward_patch_moving_avg_std])
plt.xlabel("Training-iteration")
plt.xticks([x[0], x[-1]], [0, 10**6])
plt.xlim(0, x[-1])
plt.ylabel("Reward")

# Plot distance
plt.subplot(2,1,2)
mean = np.mean(rolling_window(distance_travlled, 10), axis=-1)
var = np.var(rolling_window(distance_travlled, 10), axis=-1)
std = np.sqrt(var)
x = np.arange(0.0, distance_travlled.size, 1)

plt.plot(x, mean, color='r')
plt.fill_between(x, mean+std, mean-std, alpha=0.3, color='r')

distance_patch_moving_avg_mu = mpatches.Patch(color='r', label='Rolling average distance travelled')
distance_patch_moving_avg_std = mpatches.Patch(color='r', alpha=0.5, label='Rolling average distance travelled std')
plt.legend(handles=[distance_patch_moving_avg_mu, distance_patch_moving_avg_std])
plt.xlabel("Training-iteration")
plt.xticks([x[0], x[-1]], [0, 10**6])
plt.xlim(0, x[-1])
plt.ylabel("Distance[m]")

plt.show()

# Plot the non averaged rewards and values
plt.clf()
# Plot reward
plt.subplot(2,1,1)
x = np.arange(0.0, test_rewards.size, 1)

plt.plot(test_rewards, color='b')
reward_patch= mpatches.Patch(color='b', label='Rewards')
plt.legend(handles=[reward_patch])
plt.xlabel("Training-iteration")
plt.xticks([x[0], x[-1]], [0, 10**6])
plt.xlim(0, x[-1])
plt.ylabel("Reward")

# Plot distance
plt.subplot(2,1,2)
x = np.arange(0.0, distance_travlled.size, 1)

plt.plot(x, mean, color='r')

distance_patch= mpatches.Patch(color='r', alpha=0.5, label='Distance travelled')
plt.legend(handles=[distance_patch])
plt.xlabel("Training-iteration")
plt.xticks([x[0], x[-1]], [0, 10**6])
plt.xlim(0, x[-1])
plt.ylabel("Distance[m]")
plt.show()
print("done")
