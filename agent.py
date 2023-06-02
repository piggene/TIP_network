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
from network import MlpDQN, ModelPredictor, Encoder
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):

        #setting multiple environments
        self.env = []
        self.tau = []
        self.state = []
        self.z = []
        for i in range(len(self.env_list)):
            self.env.append(gym.make(self.env_list[i]))
            self.tau.append([])
            obs = self.env[i].reset()
            self.state.append(obs)
            self.z.append(torch.zeros(self.latent_size, device=device))
        
        self.action_size = self.env[i].action_space.n  
        self.obs_size = self.env[i].observation_space.shape[0]
        self.memory_buffer = ReplayMemory(memory_size=100000, action_size=self.action_size, obs_size=self.obs_size, latent_size=self.latent_length)               
        self.dqn_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size).to(device)
        self.predictor_network = ModelPredictor(output_size=(2+self.obs_size), input_size=(self.action_size+self.obs_size+self.latent_size)).to(device) #reward and terminal
        self.encoder_network = ModelPredictor(output_size=self.latent_size, input_size=self.tau_length*(self.obs_size+self.action_size+2)).to(device)
        
        self.optimizer_dqn = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate_dqn)
        self.optimizer_pred = optim.Adam(self.predictor_network.parameters(), lr=self.learning_rate_pred)
        self.optimizer_encoder = optim.Adam(self.encoder_network.parametrs(), lr=self.learning_rate_encd)
        
        #modify loss & criterion
        self.loss_rl = 0
        self.criterion_dqn = nn.MSELoss()
        self.criterion_pred = nn.MSELoss()
        self.criterion_encd = nn.MSELoss()

    def train(self):
        step = 0
        # A new episode
        while step < self.max_step:
            i = random.randrange(len(self.env_list))
            input_tau = torch.zeros(self.tau_length,(self.obs_size+self.action_size+2),device=device)
            for j in range(len(self.tau[i])):
                input_tau[j][:] = self.tau[i][j]
            z_hat = self.encoder_network.forward(input_tau.reshape(-1,1).squeeze())
            self.z[i] = self.z[i]*(1-self.alpha)+self.alpha*z_hat
            current_state = self.state[i]
            input_policy = torch.cat([torch.FloatTensor(current_state),self.z[i]]).to(device)

            action = random.randrange(self.action_size) if np.random.rand() < self.epsilon else \
                    torch.argmax(self.dqn_network(input_policy)).item()
            next_state, reward, terminal, truncated, info = self.env[i].step(action)

            self.memory_buffer.add(current_state, reward, action, terminal, next_state, self.tau[i], self.z[i], i)
            if not terminal:
                self.state[i] = next_state
            if terminal:
                self.state[i] = self.env[i].reset()

            ########### will modify here ###########
            if len(self.tau[i]) == self.tau_length: 
                self.tau[i].remove(0)
            new_tau = torch.cat(torch.FloatTensor(next_state), torch.FloatTensor(action)).to(device)
            new_tau = torch.cat(new_tau, torch.FloatTensor([reward, terminal])).to(device)
            self.tau[i].append(new_tau)
            #######################################

            if step > self.learning_start_rl:
                if step % self.learning_freq_rl==0:
                    self.learn_rl()
            if step > self.learning_start_encd:
                    self.learn_encd()
            
            self.epsilon_decay()
            step += 1

        for env in self.env_list:
            self.env.close()

    def learn_rl(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch = self.memory_buffer.sample(self.batch_size)
        
        y_batch = torch.FloatTensor()
        pred_batch = torch.FloatTensor()
        real_batch = torch.FloatTensor()
        for i in range(self.batch_size):
            if terminal_batch[i]:
                y_batch = torch.cat((y_batch, torch.FloatTensor([reward_batch[i]])), 0)
            else:
                j=0
                state_next = next_state_batch[i]
                one = 1
                y = one*reward_batch[i]
                while (j<self.TD_step):
                    input_policy = torch.cat([torch.FloatTensor(state_next),z_batch[i]]).to(device)
                    action_next = random.randrange(self.action_size) if np.random.rand() < self.epsilon else \
                        torch.argmax(self.dqn_network(input_policy)).item()                
                    estimate = self.predictor_network.forward(torch.cat([\
                    torch.FloatTensor(state_next),action_next]))
                    state_next_next = estimate[0:self.obs_size]
                    reward_next = estimate[-2]
                    terminal_next = estimate[-1]
                    one *= self.gamma
                    state_next = state_next_next
                    y += one*reward_next
                    if terminal_next:
                        break
                    j +=1
                if not terminal_next:
                    input_policy = torch.cat([torch.FloatTensor(state_next),z_batch[i]]).to(device)
                    y += one*self.gamma*torch.max(self.dqn_network(input_policy).to(device),dim=1)[0]
                y_batch = torch.cat((y_batch, y), 0)
            pred_batch = torch.cat((pred_batch,self.predictor_network.forward(torch.cat([torch.FloatTensor(state_batch[i],action_batch[i])]))),0) 
            real_batch = torch.cat([real_batch,torch.cat([next_state_batch[i],reward_batch[i],terminal_batch[i]])],0)

        current_state_q = torch.max(self.dqn_network(torch.FloatTensor(state_batch).to(device)), dim=1)[0]
        self.loss_rl = self.td_weight*self.criterion_dqn(current_state_q, y_batch).mean() + self.pred_weight*self.criterion_pred(pred_batch, real_batch)

        self.optimizer_dqn.zero_grad()
        self.optimizer_pred.zero_grad()
        self.loss_rl.backward()
        self.optimizer_dqn.step()
        self.optimizer_pred.step()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_minimum)

    def learn_encd(self):
        #rnn TODO
