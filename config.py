from pettingzoo.mpe import simple_spread_v2
from Environments.CoinGather import coin_gather_v0
import gym  

class Config:
    def __init__(self):

        self.replay_size = 100000 #100000 works for spread

        self.gamma = 0.9
        self.lr = 1e-4 #worked with 0.0001 too i think
        self.train_every = 10
        self.train_batches = 1
        self.update_after = 10000
        self.batch_size = 128
        self.init_epsilon = 1
        self.final_epsilon = 0.1
        self.total_steps = 3000000
        self.steps_per_epoch = 10000
        self.update_predictor_steps = self.total_steps
        self.random_steps_fraction = 0.5
        self.random_steps = int(self.random_steps_fraction * self.total_steps)
        self.target_update = 1
        self.target_update_every = 500

        self.controller = 'DecentralizedController'
        self.agent = 'DQNRewardAgent'
        self.reward_sharing = self.agent == 'DQNRewardAgent'
        if self.reward_sharing:
            assert self.controller == 'DecentralizedController'
        #self.use_state = False
        self.use_full_obs = False
        self.device = "cpu"
        envs_names = ['spread', 'coin_gather', 'gathering']
        env_name_index = 1

        self.N_agents = 2
        if env_name_index != 2:
            if env_name_index == 1:
                self.env = lambda **args: coin_gather_v0.parallel_env(**args)
                self.env_args = {'N':self.N_agents, 'max_cycles':100, 'grid_width' : 2, 'grid_height' : 9,
                            # 'levers_prob_self' : [[1, 0, 0, 0], [0.7, 0.2, 0.1, 0],
                            #                     [0, 0, 0, 1], [0, 0, 0, 1] ],
                            # 'levers_prob_other' : [[0.05, 0.9, 0.0, 0.05], [0.25, 0.05, 0.05, 0.65],
                            #                       [0, 0, 0.5, 0.5], [0, 0, 0, 1]] 
                            'levers_prob_self' : [[0, 0, 0, 1], [0, 0, 0, 1]],
                            'levers_prob_other' : [[1, 0, 0, 0], [0, 1, 0, 0]] 
                            }
                self.eval_episodes = 2
            else:
                self.env = lambda **args: simple_spread_v2.parallel_env(**args)
                self.env_args = {'local_ratio':0, 'N':self.N_agents}
                self.eval_episodes = 100
            self.rendered_env = lambda **args: self.env(**dict({'render_mode':'human'},**args))
            self.get_agents_names = lambda env: env.agents
            self.get_obs_space = lambda env, names: {name: env.observation_space(name).shape for name in names}
            self.get_act_space = lambda env, names: {name: env.action_space(name).n for name in names}
            self.step = lambda res: res
            self.hidden_layers = [32, 32]
            self.cnn = None
        else:
            self.env = lambda **args: gym.make(
                'gym_cpr_grid:CPRGridEnv-v0', 
                    **args
                )
            self.rendered_env = self.env
            self.env_args = {'n_agents':self.N_agents, 
                'grid_width':25, 
                'grid_height':7,
                'fov_squares_front':5,
                'fov_squares_side':2,
                #'global_obs':True,
                'initial_resource_probability':0.2}
            self.get_agents_names = lambda env: [i for i in range(self.N_agents)]
            self.get_obs_space = lambda env, names: {name: env.observation_space.shape for name in names}
            self.get_act_space = lambda env, names: {name: env.action_space.n for name in names}
            self.step = lambda res: (res[0], res[1], res[2], res[2], res[3])
            self.eval_episodes = 1
            self.hidden_layers = [6]
            self.cnn = [(3, 6, 3)]

        self.render_last_episode = False

        
        self.reward_hidden_layers = [32,32]
        self.N_predictors = 3
        self.perturb_prob = 0.0
        self.reward_lr = self.lr

        self.trajectory_length = 10
        self.pairs_count = 32
        self.trajectory_buffer_size = self.replay_size // self.trajectory_length // 10
        self.similarity = 1
        self.reward_weighting = 0.8

        self.save_dir = f"saved_models/{envs_names[env_name_index]}_{self.N_agents}_{self.controller}_{self.agent}"
        if self.reward_sharing:
            self.save_dir += f'_{self.perturb_prob}_{self.reward_weighting}'