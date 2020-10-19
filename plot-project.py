import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

test_rewards = []
distance_travlled = []
with open("distance_travelled.npy", 'rb') as f:
    distance_travlled = np.load(f)
with open("test_rewards.npy", 'rb') as f:
    test_rewards = np.load(f)


# Plot reward
plt.subplot(2,1,1)
plt.plot(test_rewards, color='b')
reward_patch = mpatches.Patch(color='b', label='Reward')
plt.legend(handles=[reward_patch])
plt.xlabel("Training-iteration")
plt.ylabel("Reward")

# Plot distance
plt.subplot(2,1,2)
plt.plot(distance_travlled, color='r')
distance_travelled_patch = mpatches.Patch(color='r', label='Distance travelled')
plt.legend(handles=[distance_travelled_patch])
plt.xlabel("Training-iteration")
plt.ylabel("Reward")


plt.show()
print("done")
