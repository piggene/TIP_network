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
from network import MlpDQN, MlpEmbed, CNNAdapt
from ops import *
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class Agent(AgentConfig, EnvConfig):
    def __init__(self, mode):

        self.mode = mode
        self.plotlist = []
        #setting multiple environments
        self.env = []
        self.tau = []
        self.state = []
        # self.z = []
        self.max_reward = []
        self.recent_reward = []
        self.reward_episode = []
        self.recent_len = []
        self.epsilon = []
        self.e = []
        self.task_step = []
        self.prev_step = []
        # ranges = [
        # (5.0, 15.0),
        # (0.4, 0.8),
        # (0.5, 1.5),
        # (0.05, 0.15),
        # (0.2, 0.6),
        # (0, 0.8),
        # (-0.3, 0.3),
        # (-0.3, 0.3),
        # (-0.3, 0.3),
        # (-0.8, 0.8),
        # (-0.4, 0.4),
        # (-0.4, 0.4),
        # (-0.4, 0.4),
        # ]
        ranges = [
        (5.0, 15.0),
        (0.4, 0.8),
        (0.5, 1.5),
        (0.05, 0.15),
        (1, 1),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        ]
        
        for i in range(self.env_num):
            # self.env.append(gym.make(self.env_list[i],full_action_space=True))
            task_vec = torch.tensor([torch.rand(1) * (high - low) + low for low, high in ranges],device=device).squeeze()
            task_vec_ls = task_vec.cpu().tolist()
            self.env.append(gym.make(self.env_name,physics=task_vec_ls[0:4],tasks=task_vec_ls[4:])) 
            obs, _ = self.env[i].reset()
            act_size = self.env[i].action_space.n
            obs_size = self.env[i].observation_space.shape[0]
            self.tau.append(torch.zeros([self.tau_max_length,act_size+obs_size+2],device=device))
            self.state.append(obs)
            # self.z.append(torch.zeros(self.latent_size, device=device))
            self.max_reward.append(0)
            self.recent_reward.append(0)
            self.reward_episode.append(0)
            self.recent_len.append(0)
            self.epsilon.append(1)
            self.e.append(task_vec)
            self.task_step.append(0)
            self.prev_step.append(0)

        
        self.action_size = self.env[i].action_space.n  
        self.obs_size = self.env[i].observation_space.shape[0]
        self.memory_buffer = ReplayMemory(memory_size=self.memory_size, action_size=self.action_size, obs_size=self.obs_size, taskvec_size=self.task_vec_size, num_task = self.env_num)               
        self.dqn_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.target_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        
        self.embedding_network = MlpEmbed(output_size=self.latent_size, input_size=self.task_vec_size).to(device)
        self.adaptor_network = CNNAdapt(s_size = self.obs_size, a_size = self.action_size, tau_len = self.tau_max_length, latent_size = self.latent_size)
        self.optimizer_dqn = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate_dqn)
        self.optimizer_embed = optim.Adam(self.embedding_network.parameters(), lr=self.learning_rate_embd)
        self.criterion_adaptor = nn.MSELoss() 
        self.optimizer_adaptor = optim.Adam(self.adaptor_network.parameters(), lr=self.learning_rate_adaptor) 

        #modify loss & criterion
        self.loss_rl = 0
        self.criterion_dqn = nn.MSELoss()
        
    def train(self):
        # if mode == 0 execute dqn learning stage
        if self.mode == 0:
            folder_paths = ['weights','taus']
            # Check if the folder exists, and if not, create it
            for folder_path in folder_paths:
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # Remove all files in the folder
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            try:
                step = 0
                episode = 0
                # plotlist = []
                # A new episode
                while step < self.max_step:
                    i = random.randrange(self.env_num) #TODO modify this with bandit algorithm~!

                    current_state = self.state[i]
                    current_state_torch = torch.tensor(current_state , device = device, dtype = torch.float)
                    # z = self.embedding_network(self.e[i])
                    z = self.e[i]
                    # z = torch.zeros_like(z)
                    input_policy = torch.cat([current_state_torch,z]).to(device)

                    action = random.randrange(self.action_size) if np.random.rand() < self.epsilon[i] else \
                            torch.argmax(self.dqn_network(input_policy)).item()
                    next_state, reward, terminal, _, _ = self.env[i].step(action)
                    
                    self.state[i] = next_state
                    self.reward_episode[i] = self.reward_episode[i]*self.gamma + reward 
                    
                    if self.reward_episode[i] > self.max_reward[i]:
                        self.max_reward[i] = self.reward_episode[i]

                    action_onehot = torch.zeros(self.action_size).to(device)
                    action_onehot[action] = 1    
                    next_state_torch = torch.tensor(next_state, device = device, dtype = torch.float)
                    self.tau[i][1:][:] = self.tau[i][:-1][:].clone()
                    self.tau[i][0][:-2] = torch.cat([current_state_torch,action_onehot])
                    self.tau[i][0][-2] = reward
                    self.tau[i][0][-1] = int(terminal)
                    if self.task_step[i] % self.tau_save_freq==0 and self.task_step[i] > self.tau_max_length:
                        save_filename = f"taus/tau_{i}_step{self.task_step[i]}.pt"
                        torch.save(self.tau[i], save_filename)

                    self.memory_buffer.add(current_state,reward,action,terminal,next_state,self.e[i].to('cpu'),i)
                    

                    if step > self.learning_start_rl:
                        if step % self.learning_freq_rl==0:
                            self.learn_rl()


                    if step % self.progress_freq==0:
                        print("current step: ", step)
                        for i in range(self.env_num):
                            print(f"Task {i:2d} | Max Reward: {self.max_reward[i]:6.2f} | Epsilon: {self.epsilon[i]:5.2f} | Step: {self.task_step[i]:5d} | Recent Reward: {self.recent_reward[i]:6.2f} | Episode Length: {self.recent_len[i]:5d} | Loss Sum: {sum(self.plotlist):3.3f}")
                        print("==================================================================")
                        self.plotlist = []

                    # plotlist.append(self.max_reward) 
                    if self.max_reward[i] > 20 and step > self.learning_start_rl:            
                        self.epsilon_decay(i)
                    step += 1
                    self.task_step[i] += 1
                    if terminal:
                        self.state[i], _ = self.env[i].reset()
                        # print("step: ",step,"env: ",i,"epi_reward: ",self.reward_episode[i],"epsilon: ",self.epsilon[i])
                        self.recent_reward[i] = self.reward_episode[i]
                        self.recent_len[i] = self.task_step[i]-self.prev_step[i]
                        self.reward_episode[i] = 0
                        self.prev_step[i] = self.task_step[i]
                        episode += 1
                        if episode % self.target_freq == 0:
                            self.target_network.load_state_dict(self.dqn_network.state_dict())

            except KeyboardInterrupt:
                print("Training Interrupted. Saving Model weights...")
                torch.save(self.dqn_network.state_dict(),f"weights/dqn_weight.pt")
                torch.save(self.embedding_network.state_dict(),f"weights/embedding_weight.pt")
                for i in range(self.env_num):
                    save_filename = f"taus/taskvec_{i}.pt"
                    torch.save(self.e[i],save_filename)
            
            print("Training finished. Saving final model weights...")
            torch.save(self.dqn_network.state_dict(),f"weights/dqn_weight.pt")
            torch.save(self.embedding_network.state_dict(),f"weights/embedding_weight.pt")
            
            for env in self.env:
                env.close()

        else: 
            self.learn_adaptor()

    class CustomDataset(Dataset):
        def __init__(self, folder_path, z):
            self.folder_path = folder_path
            self.z = z
            self.file_list = [f for f in os.listdir(folder_path) if f.startswith('tau')]

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            file_name = self.file_list[idx]
            file_path = os.path.join(self.folder_path, file_name)

            tau_ij = torch.load(file_path)
            class_i = int(file_name.split('_')[1])  # Extract the class from the folder name
            z_i = self.z[class_i]

            return tau_ij, z_i       


    def learn_rl(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, e_batch, task_batch = self.memory_buffer.sample(self.batch_size_rl)
        
        y_batch = torch.FloatTensor().to(device)
        pred_batch = torch.FloatTensor().to(device)
        real_batch = torch.FloatTensor().to(device)
        q_val_batch = torch.FloatTensor().to(device)
        for i in range(self.batch_size_rl):
            # z = self.embedding_network.forward(torch.FloatTensor(e_batch[i]).to(device))
            z = torch.FloatTensor(e_batch[i]).to(device)
            if terminal_batch[i]:
                y_batch= torch.cat([y_batch,torch.FloatTensor([reward_batch[i]]).to(device)])
            else:
                state_next = next_state_batch[i]
                y = reward_batch[i]
                input_policy = torch.cat([torch.FloatTensor(state_next).to(device),z]).to(device)
                # print(self.dqn_network(input_policy).to(device))
                y += self.gamma*torch.max(self.target_network(input_policy).to(device))
                y_batch = torch.cat([y_batch, y.reshape(1)])
            input_policy = torch.cat([torch.FloatTensor(state_batch[i]).to(device),z]).to(device)
            q_val_batch = torch.cat([q_val_batch,self.dqn_network(input_policy).to(device)[int(action_batch[i].item())].reshape(1)]) 

        self.loss_rl = self.criterion_dqn(q_val_batch, y_batch).mean()
        self.plotlist.append(self.loss_rl.item())
        self.optimizer_dqn.zero_grad()
        # self.optimizer_embed.zero_grad()
        self.loss_rl.backward()
        self.optimizer_dqn.step()
        # self.optimizer_embed.step()

    def epsilon_decay(self, i):
        self.epsilon[i] *= self.epsilon_decay_rate
        self.epsilon[i] = max(self.epsilon[i], self.epsilon_minimum)

    def learn_adaptor(self) :

        state_dict = torch.load('weights/embedding_weight.pt')
        self.embedding_network.load_state_dict(state_dict)
        z = []
        folder_path = 'taus' 
        # Iterate over the files in the folder
        for i in range(self.env_num):  
            file_path = os.path.join(folder_path, f'taskvec_{i}.pt')

            # Check if the file exists
            if os.path.exists(file_path):
                # Load the vector using torch.load
                e_i = torch.load(file_path)
                # Append the loaded vector to the list
                z.append(self.embedding_network(e_i).to(device))
            else:
                print(f"File not found: {file_path}")
        
        for param in self.embedding_network.parameters():
            param.requires_grad = False

        dataset = CustomDataset(folder_path, z)
        batch_size = self.batch_size_adaptor
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        num_epochs = self.epoch_adaptor  

        for epoch in range(num_epochs):
            for batch_taus, batch_z in dataloader:
                self.optimizer_adaptor.zero_grad()

                # Forward pass
                predicted_z = self.adaptor_network(batch_taus)

                # Compute the loss
                loss = self.criterion_adaptor(predicted_z, batch_z)

                # Backward pass
                loss.backward()

                # Update the weights
                self.optimizer_adaptor.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


        