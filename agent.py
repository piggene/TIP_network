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
from network import MlpDQN, ModelPredictor, Encoder_rnn, Decoder_rnn, Seq2Seq
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):

        #setting multiple environments
        self.env = []
        self.tau = []
        self.state = []
        self.z = []
        self.max_reward = []
        self.reward_episode = []
        self.epsilon = []
        for i in range(len(self.env_list)):
            self.env.append(gym.make(self.env_list[i],full_action_space=True))
            self.tau.append([])
            obs = self.env[i].reset()
            self.state.append(obs)
            self.z.append(torch.zeros(self.latent_size, device=device))
            self.max_reward.append(0)
            self.reward_episode.append(0)
            self.epsilon.append(1)

        
        self.action_size = self.env[i].action_space.n  
        self.obs_size = self.env[i].observation_space.shape[0]
        self.memory_buffer = ReplayMemory(memory_size=100000, action_size=self.action_size, obs_size=self.obs_size, latent_size=self.latent_size, num_task = len(self.env_list))               
        self.dqn_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.predictor_network = ModelPredictor(output_size=(2+self.obs_size), input_size=(self.action_size+self.obs_size+self.latent_size)).to(device) #reward and terminal


        input_dim = self.obs_size+self.action_size+2
        output_dim = self.obs_size+self.action_size+2
        hid_dim = self.latent_size

        self.enc = Encoder_rnn(input_dim = input_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.enc_dropout)
        self.dec = Decoder_rnn(output_dim = output_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.dec_dropout)
        self.seq2seq = Seq2Seq(self.enc, self.dec, device).to(device)

        
        self.optimizer_dqn = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate_dqn)
        self.optimizer_pred = optim.Adam(self.predictor_network.parameters(), lr=self.learning_rate_pred)
        self.optimizer_encoder = optim.Adam(self.seq2seq.parameters(), lr=self.learning_rate_encd)
        
        #modify loss & criterion
        self.loss_rl = 0
        self.loss_encd = 0
        self.criterion_dqn = nn.MSELoss()
        self.criterion_pred = nn.MSELoss()
        self.criterion_encd = nn.MSELoss()

    def train(self):
        step = 0
        plotlist = []
        # A new episode
        while step < self.max_step:
            i = random.randrange(len(self.env_list))


            #Extracting latent representation from tau
            if len(self.tau[i])==0:
                z_hat = self.z[i]
            
            else:
                input_tau = torch.zeros(len(self.tau[i]), (self.obs_size+self.action_size+2),device=device)
                for j in range(len(self.tau[i])):
                    input_tau[j][:] = self.tau[i][j]
                z_hat = self.seq2seq.encoder(input_tau.unsqueeze(1)).squeeze(0).squeeze(0)


            #self.z[i] = self.z[i]*(1-self.alpha)+self.alpha*z_hat
            #self.alpha_decay()
            self.z[i]= z_hat
            current_state = self.state[i]
            input_policy = torch.cat([torch.FloatTensor(current_state),self.z[i]]).to(device)

            action = random.randrange(self.action_size) if np.random.rand() < self.epsilon[i] else \
                    torch.argmax(self.dqn_network(input_policy)).item()
            next_state, reward, terminal, info = self.env[i].step(action)
            if len(self.tau[i])!=0:
                self.memory_buffer.add(current_state, reward, action, terminal, next_state, self.tau[i][-1], self.z[i], i)
            if not terminal:
                self.state[i] = next_state
                self.reward_episode[i] = self.reward_episode[i]*self.gamma + reward 
            if terminal:
                self.state[i] = self.env[i].reset()
                self.reward_episode[i] = 0

            if self.reward_episode[i] > self.max_reward[i]:
                self.max_reward[i] = self.reward_episode[i]

            if len(self.tau[i]) == self.tau_max_length: 
                self.tau[i].remove(0)
            action_onehot = torch.zeros(self.action_size).to(device)
            action_onehot[action] = 1    
            new_tau = torch.cat([torch.FloatTensor(next_state), torch.FloatTensor(action_onehot)]).to(device)
            new_tau = torch.cat([new_tau, torch.FloatTensor([reward, terminal])]).to(device)
            self.tau[i].append(new_tau)


            if step > self.learning_start_rl:
                if step % self.learning_freq_rl==0:
                    self.learn_rl()
            if step > self.learning_start_encd:
                if step % self.learning_freq_encd==0:
                    self.learn_encd()

            if step % self.progress_freq==0:
                print("current step: ", step)
                for i in range(len(self.env_list)):
                    print(self.env_list[i] + " max reward: ", self.max_reward[i]) 
                print("==================================================================")

            plotlist.append(self.max_reward)            
            self.epsilon_decay(i)
            step += 1


        for env in self.env_list:
            self.env.close()

        #self.plot_graph(plotlist)

    def learn_rl(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch = self.memory_buffer.sample(self.batch_size_rl)
        
        y_batch = torch.FloatTensor().to(device)
        pred_batch = torch.FloatTensor().to(device)
        real_batch = torch.FloatTensor().to(device)
        q_val_batch = torch.FloatTensor().to(device)
        for i in range(self.batch_size_rl):
            if terminal_batch[i]:
                y_batch= torch.cat([y_batch,torch.FloatTensor([reward_batch[i]])])
            else:
                j=0
                state_next = next_state_batch[i]
                one = 1
                y = one*reward_batch[i]
                while (j<self.TD_step):
                    input_policy = torch.cat([torch.FloatTensor(state_next),torch.FloatTensor(z_batch[i])]).to(device)
                    action_idx = random.randrange(self.action_size) if np.random.rand() < self.epsilon[int(task_batch[i].item())] else \
                        torch.argmax(self.dqn_network(input_policy)).item() 
                    action_next = torch.zeros(self.action_size).to(device)
                    action_next[action_idx] = 1              
                    estimate = self.predictor_network.forward(torch.cat([\
                    torch.FloatTensor(state_next),action_next,torch.FloatTensor(z_batch[i])]))
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
                    input_policy = torch.cat([torch.FloatTensor(state_next),torch.FloatTensor(z_batch[i])]).to(device)
                    y += one*self.gamma*torch.max(self.dqn_network(input_policy).to(device),dim=1)[0]
                y_batch = torch.cat([y_batch, y.reshape(1)])
            action_next = torch.zeros(self.action_size)
            action_next[int(action_batch[i].item())] = 1
            pred_batch = torch.cat([pred_batch,self.predictor_network.forward(torch.cat([torch.FloatTensor(state_batch[i]), action_next, torch.FloatTensor(z_batch[i])]))],0) 
            real_batch = torch.cat([real_batch,torch.cat([torch.FloatTensor(next_state_batch[i]),torch.FloatTensor([reward_batch[i],terminal_batch[i]])])])
            input_policy = torch.cat([torch.FloatTensor(state_batch[i]),torch.FloatTensor(z_batch[i])]).to(device)
            q_val_batch = torch.cat([q_val_batch,self.dqn_network(input_policy).to(device)[int(action_batch[i].item())].reshape(1)]) 

        self.loss_rl = self.td_weight*self.criterion_dqn(q_val_batch, y_batch).mean() + \
        self.pred_weight*self.criterion_pred(pred_batch, real_batch).mean()

        self.optimizer_dqn.zero_grad()
        self.optimizer_pred.zero_grad()
        self.loss_rl.backward()
        self.optimizer_dqn.step()
        self.optimizer_pred.step()

    def epsilon_decay(self, i):
        self.epsilon[i] *= self.epsilon_decay_rate
        self.epsilon[i] = max(self.epsilon[i], self.epsilon_minimum)

    def alpha_decay(self):
        return 0

    def learn_encd(self):

        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch = self.memory_buffer.sample(self.batch_size_seq)

        tau_batch_torch = nn.utils.rnn.pad_sequence(tau_batch)
        outputs, _ = self.seq2seq(tau_batch_torch, tau_batch_torch, teacher_forcing_ratio=self.teacher_forcing_ratio)
        ## input = [src len, batch size, input dim]
        self.loss_encd = self.criterion_encd(outputs, tau_batch_torch)

        self.optimizer_encoder.zero_grad()
        self.loss_encd.backward()
        self.optimizer_encoder.step() 

    def plot_graph(self, plotlist):
        return 0


        