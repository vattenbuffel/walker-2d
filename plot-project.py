import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

test_rewards = []
distance_travlled = []
with open("distance_travelled.npy", 'rb') as f:
    distance_travlled = np.load(f)
with open("test_rewards.npy", 'rb') as f:
    test_rewards = np.load(f)


# Plot reward
plt.subplot(2,1,1)
plt.plot(test_rewards[0:-10], color='b', alpha=0.5)
plt.plot(moving_average(test_rewards, n=10), color='b')
reward_patch_moving_avg = mpatches.Patch(color='b', label='Reward-moving average')
reward_patch = mpatches.Patch(color='b', alpha=0.5, label='Reward')
plt.legend(handles=[reward_patch_moving_avg, reward_patch])
plt.xlabel("Training-iteration")
plt.ylabel("Reward")

# Plot distance
plt.subplot(2,1,2)
plt.plot(moving_average(distance_travlled, n=10), color='r')
plt.plot(distance_travlled[0:-10], color='r', alpha=0.5)
distance_travelled_patch_moving_avg = mpatches.Patch(color='r', label='Distance travelled-moving average')
distance_travelled_patch = mpatches.Patch(color='r', label='Distance travelled', alpha=.5)
plt.legend(handles=[distance_travelled_patch, distance_travelled_patch_moving_avg])
plt.xlabel("Training-iteration")
plt.ylabel("Reward")


plt.show()
print("done")
