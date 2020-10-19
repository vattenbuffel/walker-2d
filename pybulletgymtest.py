import gym
import pybulletgym

env_name = "InvertedPendulumPyBulletEnv-v0"
env = gym.make(env_name)
env.render()
env.reset()

ethae = 5

env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])
env.step([1])