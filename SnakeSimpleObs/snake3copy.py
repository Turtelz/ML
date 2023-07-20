import gym
from gym import spaces
import keyboard
import pygame
import numpy as np
from random import randint
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
import torch
import torch.nn as nn
testEnv = True


class Snake(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, size=20):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.np_random = np.random
        self.maxFrames = 25
        self.frame = 0
        self.growBy = 5
        self.body = []
        self.direction = 3
        self.die = 100
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        
        
        self.observation_space = spaces.Dict({
                                            "FoodX": spaces.Discrete(3), # [0,4]
                                            "FoodY": spaces.Discrete(3),
                                            "Danger":spaces.Box(0,self.size,shape=(4,),dtype=int), #[-1, self.size] distance to nearest wall or snake bod
                                            "Direction":spaces.Discrete(4) # curent direction moving
                                            })
        '''
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "body": spaces.Box(low=0,high=size-1,shape=((size * size) * 2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        '''
        #print(self.observation_space)
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # distance to danger left up right down (danger is closest wall or snake part)
        # distance to food left right up down
        # [0] is x, [1] is y
        
        head = self._agent_location
        food = self._target_location
        dxFood = head[0] - food[0]
        dyFood = head[1] - food[1]
        
        # 0 is down, 1 is right, 2 is left, 3 is up 
        fx = 2
        if dxFood < 0:
            fx = 0
        elif dxFood > 0:
            fx = 1
            
        fy = 2
        if dyFood > 0:
            fy = 1
        elif dyFood <0:
            fy = 0

        
        
            
        # raycast kinda of left right down up of neareast danger    
        LeftRight = []
        left = []
        right = []
        for bod in self.body:  # right left danger
            if bod[1] == head[1]:
                LeftRight.append(head[0] - bod[0])
        left = list(filter(lambda x : x > 0, LeftRight))
        right = list(filter(lambda x : x < 0, LeftRight)) # neg
        closestLeft = -1
        closestRight = -1
        
        if len(left) != 0:
            closestLeft = min(left)
        else:
            closestLeft = head[0]
        if len(right) != 0:
            closestRight = min(list(map(lambda x: abs(x),right)))
        else:
            closestRight = self.size - head[0]
            
        UpDown = []
        for bod in self.body:  # right left danger
            if bod[0] == head[0]:
                UpDown.append(head[1] - bod[1])
        
        up = []
        down = []
        up = list(filter(lambda x : x > 0, UpDown))
        down = list(filter(lambda x : x < 0, UpDown)) # neg
        
        
        
        closestUp = -1
        closestDown = -1
        if len(up) != 0:
            closestUp = min(up)
        else:
            closestUp = head[1]
        if len(down) != 0:    
            closestDown = min(list(map(lambda x: abs(x),down)))
        else:
            closestDown = self.size - head[1]
                
        d ={"FoodX": fx,"FoodY": fy,"Danger": np.array([closestDown,closestRight,closestLeft,closestUp]),"Direction": self.direction}
        #print(d)
        return d


    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }


    def moveFood(self):
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.randint(
                0, self.size, size=2, dtype=int
            )
    def reset(self, seed=None, options=None):
        self.die = self.size * self.size
        self.body = []
        self.growBy = 5
        self.frame = 0
        # We need the following line to seed self.np_random
        #super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.randint(0, self.size, size=2, dtype=int)
        self._agent_location = np.array([(int)(self.size/2),(int)(self.size/2)])
        #print(self._agent_location)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([(int)(self.size/2),(int)(self.size/2) - 1])
        #self.moveFood()

        observation = self._get_obs()
        # info = self._get_info()  stablebaseline does not want info

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if self.direction + action != 3:
            self.direction = action
        
        
        
        
        reward = 0
        direction = self._action_to_direction[self.direction]
        # We use `np.clip` to make sure we don't leave the grid
       # print(self.frame,self._agent_location)
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
       # print(self.frame,self._agent_location)
        
        
        # intect with body
        intercect = False
        for bod in self.body:
            if np.array_equal(self._agent_location, bod):
                intercect = True
                reward = -10
                break
        terminated = intercect
        foodEaten = np.array_equal(self._agent_location, self._target_location)
        
        
        
        self.body.append(self._agent_location)
        
        if self.growBy > 0:
            self.growBy -= 1
        else:
            self.body.pop(0)
        
        # if snake gets closer reward increases, further reward decreases
        '''
        if abs((self._agent_location[0] + direction[0]) - self._target_location[0]) < abs(self._agent_location[0] - self._target_location[0]):
            reward = 1
        if abs((self._agent_location[0] + direction[0]) - self._target_location[0]) > abs(self._agent_location[0] - self._target_location[0]):
            reward = -1
            
        if abs((self._agent_location[1] + direction[1]) - self._target_location[1]) < abs(self._agent_location[1] - self._target_location[1]):
            reward = 1
        if abs((self._agent_location[1] + direction[1]) - self._target_location[1]) > abs(self._agent_location[1] - self._target_location[1]):
            reward = -1
        '''
        
        if foodEaten:
            self.moveFood()
            self.growBy += 5
            reward += 1
            self.die += self.size * self.size
        if self.die < 0:
            terminated = True
        else:
            self.die -= 1
        

        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )       
        # drawing the body 
        for bod in self.body:
            pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * bod,
                (pix_square_size, pix_square_size),
            ),
        )
            
            
        # Now we draw the agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )



        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    pass
env = Snake(size=20)
check_env(env)
print("passed Check_env")

model = DQN("MultiInputPolicy", env, verbose=1)
save_directory = './models/snake/'
numberOfCheckpoints = 10
totalTimeSteps = 50000000
checkpoint_callback = CheckpointCallback(save_freq=totalTimeSteps/numberOfCheckpoints, save_path=save_directory,
                                         name_prefix='SimpleSnake')

# notes for next time: todo
# add more input paramaters and change it to a cnn instead of a mlp 
# potetian input paramaders 
#   tail location
#   growby 
#   

#model.learn(total_timesteps=totalTimeSteps,callback=checkpoint_callback) 

model = model.load("ML/models/snake/SimpleSnake_50000000_steps")

env = Snake(render_mode='human', size=20)
obs = env.reset()

totalReward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    #action = 0

    obs, reward, done, info = env.step(action)
    #print(obs)
    #print(obs["board"].reshape(10,10))
    totalReward += reward
    env.render()
    
    if done:
        print(f"Died with {totalReward}")
        obs = env.reset()
        totalReward = 0
    if keyboard.is_pressed('q'):
        print("Quiting")
        break
env.close()