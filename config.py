class AgentConfig:
    
    #Latent Vector z's length
    latent_size = 1 #128
    #Length of memory buffer tau
    tau_max_length = 50
    pred_weight = .5
    td_weight = .5

    TD_step = -6

    learning_start_dqn = 100
    learning_freq_dqn = 30
    learning_start_encd = 20
    learning_freq_encd = 10
    learning_freq_pred = 10
    learning_start_pred = 100


    # Learning
    gamma = 0.99
    memory_size = 1000000
    batch_size_rl_dqn = 32
    batch_size_rl_pred = 16
    batch_size_seq = 16

    epsilon_minimum = 0.0005 #0.0005
    epsilon_decay_rate = 0.9999
    
    weight_decay_dqn = 0.9999
    weight_decay_encd = 0.9999
    weight_decay_pred = 0.9999
    
    alpha_minimum = 0.02
    alpha_decay_rate = 0.9995
    
    learning_rate_dqn = 0.0005
    learning_rate_pred = 0.0005
    learning_rate_encd = 0.0005
    
    target_update_freq = 200

    max_step = 4000000      # 40M steps max

    # RNN
    n_layers =  1 #Dont Change!! Fuck U
    enc_dropout = 0
    dec_dropout = 0
    teacher_forcing_ratio = 0.5


    #Progress 
    progress_freq = 100


class EnvConfig:
    env_list = ['CartPole-v0']
    # env_list = ['CartPoleStay-v0', 'CartPoleStay-v0']
    # env_list = ['Assault-ram-v4','BeamRider-ram-v4','AirRaid-ram-v4','Enduro-ram-v4','Skiing-ram-v4']
