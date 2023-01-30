# import os
# cwd = os.getcwd()
# print(cwd)
# os.chdir(cwd+'/pybullet-gym')
# cwd = os.getcwd()
# print(cwd)

import gym
import pybulletgym,shadowhand_gym
import shadowhand_gym.pybullet
import numpy as np
import moviepy.video.io.ImageSequenceClip
models = ["Reacher-v2", "Reacher-v4",
    "Pusher-v2", "Pusher-v4",
    "InvertedPendulum-v2", "InvertedPendulum-v4",
    "InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4",
    "HalfCheetah-v2", "HalfCheetah-v3", "HalfCheetah-v4",
    "Hopper-v2", "Hopper-v3", "Hopper-v4",
    "Swimmer-v2", "Swimmer-v3", "Swimmer-v4",
    "Walker2d-v2", "Walker2d-v3", "Walker2d-v4",
    "Ant-v2", "Ant-v3", "Ant-v4",
    "HumanoidStandup-v2", "HumanoidStandup-v4",
    "Humanoid-v2", "Humanoid-v3", "Humanoid-v4"]

shadowhand_gym_models = ['ShadowHandReach-v1','ShadowHandBlock-v1']
#"ShadowHandBlock-v1"
env = gym.make('ShadowHandReach-v1')
state = env.reset()

done = False
width=256
height= 256
images = [env.render("rgb_array",width=width,height=height)] #
while not done:
    # Random action
    action = env.action_space.sample()
    state, reward, done, info = env.step(action) #
    images.append(env.render("rgb_array",width=width,height=height))


env.close()
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=15)
clip.write_videofile('my_video.mp4')