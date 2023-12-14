class AgentConfig:
    
    #Latent Vector z's length
    latent_size = 13#128
    task_vec_size = 13
    #Length of memory buffer tau
    tau_max_length = 50
    tau_save_freq = 25

    learning_start_rl = 400
    learning_freq_rl = 1
    target_freq = 200


    # Learning
    gamma = 0.99
    memory_size = 4000000
    batch_size_rl = 32


    epsilon_minimum = 0.01
    epsilon_decay_rate = 0.9999
    learning_rate_dqn = 0.001
    learning_rate_embd = 0.001


    learning_rate_adaptor = 0.001
    epoch_adaptor = 50
    batch_size_adaptor = 32

    max_step = 4000000    # 40M steps max


    #Progress 
    progress_freq = 1000


class EnvConfig:
    env_name = 'DFTPole'#,'BeamRider-ram-v4','AirRaid-ram-v4','Enduro-ram-v4','Skiing-ram-v4']
    env_num = 1
