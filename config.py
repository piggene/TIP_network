class AgentConfig:
    
    #Latent Vector z's length
    latent_length = 128
    #Length of memory buffer tau
    tau_length = 50
    alpha = 0.02 # for controlling z incrementally
    pred_weight = .5
    td_weight = .5

    learning_start_rl = 1000
    learning_freq_rl = 14
    learning_start_encd = 20


    # Learning
    gamma = 0.99
    memory_size = 1000000
    batch_size = 32

    epsilon = 1
    epsilon_minimum = 0.05
    epsilon_decay_rate = 0.9999
    learning_rate_dqn = 0.001
    learning_rate_pred = 0.001
    learning_rate_encd = 0.001

    max_step = 40000000       # 40M steps max


class EnvConfig:
    env_list = ['CartPole-v0','CartPole-v0','Cartpole-v0','Cartpole-v0','Cartpole-v0']
