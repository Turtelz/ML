import gym 
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n
'''
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        action = random.choice([0,1])
        n_state, reward, done, truncated ,info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))'''
    

def build_model(states, actions):
    model = tf.keras.Sequential([keras.layers.Flatten(input_shape=(1,states)),
                                 keras.layers.Dense(24,activation='relu'),
                                 keras.layers.Dense(24,activation='relu'),
                                 keras.layers.Dense(actions,activation = 'linear')                             
                                 ])
    ''' model = tf.keras.Sequential()
    model.add(tf.keras.Flatten(input_shape=(1,states)))
    model.add(tf.keras.Dense(24, activation='relu'))
    model.add(tf.keras.Dense(24, activation='relu'))
    model.add(tf.keras.Dense(actions, activation='linear'))'''
    return model

model = build_model(states, actions)
model.summary()

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
dqn = build_agent(model, actions)
dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
#dqn.compile(optimizer='adam', metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))