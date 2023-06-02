import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AgentConfig, EnvConfig
from memory import ReplayMemory
from network import MlpPolicy, ModelPredictor, Encoder
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):


        #setting multiple environments
        self.env = []
        self.tau = []
        self.state = []
        self.episode = []
        self.z = []
        for i in range(len(self.env_list)):
            self.env.append(gym.make(self.env_list[i]))
            self.tau.append([])
            obs = self.env[i].reset()
            self.state.append(obs)
            self.episode.append(0)
            self.z.append(torch.zeros(self.latent_size, device=device))
        
        self.action_size = self.env[i].action_space.n  
        self.obs_size = self.env[i].observation_space.n
               
        self.policy_network = MlpPolicy(action_size=self.action_size, input_size=self.obs_size).to(device)
        self.target_policy = MlpPolicy(action_size=self.action_size, input_size=self.obs_size).to(device)
        self.predictor_network = ModelPredictor(output_size=(2+self.obs_size), input_size=(self.action_size+self.obs_size+self.latent_size)).to(device) #reward and terminal
        self.encoder_network = ModelPredictor(output_size=self.latent_size, input_size=self.tau_length*(self.obs_size+self.action_size+2)).to(device)
        
        self.optimizer_rl = optim.Adam(self.policy_network.parameters()+self.predictor_network.parametrs(), lr=self.learning_rate_pol)
        self.optimizer_encoder = optim.Adam(self.encoder_network.parametrs(), lr=self.learning_rate_encd)
        
        #modify loss & criterion
        self.loss_rl = 0
        self.criterion_rl = nn.MSELoss()
        self.criterion_encd = nn.MSELoss()

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, truncated, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        episode = 0
        step = 0
        reward_history = []

        # A new episode
        while step < self.max_step:
            i = randint(0, len(self.env_list)-1)
            input_tau = torch.zeros(self.tau_length,(self.obs_size+self.action_size+2),device=device)
            for j in range(len(self.tau[i]))
                input_tau[j][:] = tau[i][j]
            z_hat = encoder_network.forward(input_tau.reshape(-1,1).squeeze())
            self.z[i] = self.z[i]*(1-self.alpha)+self.alpha*z_hat
            current_state = self.state[i]
            input_policy = torch.cat([torch.FloatTensor(current_state),self.z[i]]).to(device)

            action = random.randrange(self.action_size) if np.random.rand() < self.epsilon else \
                    torch.argmax(self.policy_network(input_policy)).item()
            next_state, reward, terminal, truncated, info = self.env[i].step(action)
            target_val = reward
            one = 1
            state_next = next_state
            while (j<self.TD_step):
                input_policy = torch.cat([torch.FloatTensor(state_next),self.z[i]]).to(device)
                state_next_next, reward_next, terminal_next = predictor_network.forward(torch.cat([\
                torch.FloatTensor(state_next),torch.argmax(self.policy_network(input_policy))]))
                one *= self.gamma
                state_next = state_next_next
                target_val += one*reward_next
            
            if not terminal_next:
                input_policy = torch.cat([torch.FloatTensor(state_next),self.z[i]]).to(device)
                target_val += one*self.gamma*torch.max(self.policy_network(input_policy))


            self.memory.add(current_state, reward, action, terminal, next_state, self.tau[i], self.z[i], target_val)
            if not terminal:
                self.state[i] = next_state
            if terminal:
                self.state[i] = self.env[i].reset()

            if len(self.tau[i]) == self.tau_length: 
                self.tau[i].remove(0)
            new_tau = torch.cat(torch.FloatTensor(next_state), torch.FloatTensor(action)).to(device)
            new_tau = torch.cat(new_tau, torch.FloatTensor([reward, terminal])).to(device)
            self.tau[i].append(new_tau)


            if step > self.learning_start_rl:
                if step % self.learning_freq_rl==0:
                    learn_rl()
            if step > self.learning_start_encd:
                    learn_encd()

            step += 1

        for env in env_list:
            self.env.close()

    def learn_rl(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, target_val_batch = self.memory.sample(self.batch_size)

        current_state_q = torch.max(self.policy_network(torch.FloatTensor(state_batch).to(device)), dim=1)[0]

        self.loss_rl = self.criterion_rl(current_state_q, target_val_batch).mean()

        self.optimizer_rl.zero_grad()
        self.loss_rl.backward()
        self.optimizer_rl.step()

    def learn_encd(self):
        
