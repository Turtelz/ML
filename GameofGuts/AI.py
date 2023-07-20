


from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
from gym import spaces, Env
from gym.spaces import Box
import keyboard
import pygame
import numpy as np
from random import randint
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
import random
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class GutsGame(gym.Env):
    def __init__(self):
        # Actions we can take bank, or roll again
        self.action_space = spaces.Discrete(2)
        #  bank and unbanked
        self.observation_space = spaces.Box(0,150,shape=(2,),dtype=np.uint8)
        # Set start value
        self.bank = 0
        self.unbank = 0
        # Set turn
        self.turn = 1000
        
    def step(self, action):
        # Apply action
        # 0: bank
        # 1: roll again
        
        reward = 0
        if action == 1:
            diceRoll1 = random.randint(1, 6)
            diceRoll2 = random.randint(1, 6)
            #print(diceRoll1,diceRoll2)
            if diceRoll1 == 6 and diceRoll2 == 6:
                #reward += -(self.unbank + self.bank)
                self.unbank = 0
                self.bank = 0
            elif diceRoll1 == 6 or diceRoll2 == 6:
                #reward += -self.unbank
                self.unbank = 0
            else:
                self.unbank += (diceRoll1 + diceRoll2)
        else:
            self.bank += self.unbank
            self.unbank = 0
        
        # Reduce turn
        self.turn -= 1 
        
        # Calculate reward
        #reward += self.bank + self.unbank
        
        #reward /= 100
        #reward *= 0
        # Check if timer is done or if player won
        if self.turn <= 0 or self.bank >= 100:
            reward += self.turn
            done = True
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        obs = np.array([self.bank,self.unbank]).astype(np.uint8)
        # Return step information
        return obs, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        # Reset start value
        self.bank = 0
        self.unbank = 0
        obs = np.array([self.bank,self.unbank]).astype(np.uint8)
        #print(np.shape(obs))
        # Reset time
        self.turn = 1000
        return obs
env = GutsGame()
check_env(env)
print("passed Check_env")

device = "cpu"

model = A2C("MlpPolicy", env, device=device, verbose=1)

model = model.load("ML/GameofGuts/GutBot_50000_stepsA2C")


def predict(bankedPoints,unbankedPoints):
    print(model.predict([bankedPoints,unbankedPoints], deterministic=True))
    
predict(0,15)
