import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

test_rewards = np.array([1,2,3])
distance_travlled = np.array([4,5,7])


plt.plot(test_rewards, color='b')
reward_patch = mpatches.Patch(color='b', label='Reward')

plt.plot(distance_travlled, color='r')
distance_travelled_patch = mpatches.Patch(color='r', label='Distance travelled')

plt.legend(handles=[reward_patch, distance_travelled_patch])
plt.xlabel("Training-iteration")
plt.ylabel("Value")

plt.show()
balle = 5
print("done")
