class AgentConfig:
    
    #Latent Vector z's length
    latent_size = 128
    #Length of memory buffer tau
    tau_max_length = 5000000000000000000
    alpha = 0.02 # for controlling z incrementally
    pred_weight = .5
    td_weight = .5

    TD_step = 7

    learning_start_rl = 1000
    learning_freq_rl = 30
    learning_start_encd = 20
    learning_freq_encd = 10


    # Learning
    gamma = 0.99
    memory_size = 1000000
    batch_size_rl = 32
    batch_size_seq = 16

    epsilon_minimum = 0.05
    epsilon_decay_rate = 0.9999
    learning_rate_dqn = 0.001
    learning_rate_pred = 0.001
    learning_rate_encd = 0.001

    max_step = 4000000      # 40M steps max

    # RNN
    n_layers =  1 #Dont Change!! Fuck U
    enc_dropout = 0
    dec_dropout = 0
    teacher_forcing_ratio = 0.5


    #Progress 
    progress_freq = 100


class EnvConfig:
    env_list = ['Assault-ram-v4','BeamRider-ram-v4','AirRaid-ram-v4','Enduro-ram-v4','Skiing-ram-v4']
