class AgentConfig:
    
    #Latent Vector z's length
    latent_size = 12#128
    task_vec_size = 13
    #Length of memory buffer tau
    tau_max_length = 50
    tau_save_freq = 25

    learning_start_rl = 1000
    learning_freq_rl = 30



    # Learning
    gamma = 0.99
    memory_size = 3000
    batch_size_rl = 32


    epsilon_minimum = 0.01
    epsilon_decay_rate = 0.9995
    learning_rate_dqn = 0.001
    learning_rate_embd = 0.001

    max_step = 4000000    # 40M steps max


    #Progress 
    progress_freq = 100


class EnvConfig:
    env_name = 'DFTPole'#,'BeamRider-ram-v4','AirRaid-ram-v4','Enduro-ram-v4','Skiing-ram-v4']
    env_num = 40
