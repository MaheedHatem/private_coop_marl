from pettingzoo.mpe import simple_spread_v2
from Environments.CoinGather import coin_gather_v0
import lbforaging
import gym  

import yaml
import json

def get_config(config_file):
    with open(f"Configs/{config_file}.yaml", "r") as f:
        data = yaml.safe_load(f)
    return Config(data)
        
    

class Config:
    def __init__(self, data):


        self.controller = data['controller']
        self.agent = data['agent']
        self.reward_sharing = self.agent == 'DQNRewardAgent'
        self.true_reward = data['true_reward']
        if self.true_reward:
            assert self.controller == 'DecentralizedController' and self.agent == 'DQNAgent'
        if self.reward_sharing:
            assert self.controller == 'DecentralizedController'
        #self.use_state = False
        self.use_full_obs = data['use_full_obs']
        self.device = data['device']

        self.replay_size = data['replay_size'] #100000 works for spread
        self.gamma = data['gamma']
        self.lr = data['lr'] #worked with 0.0001 too i think
        self.train_every = data['train_every']
        self.train_batches = data['train_batches']
        self.update_after = data['update_after']
        self.batch_size = data['batch_size']
        self.total_steps = data['total_steps']
        self.steps_per_epoch = data['steps_per_epoch']
        self.update_predictor_steps = self.total_steps
        if self.agent == 'ACAgent':
            self.value_coef = data['value_coef']
            self.entropy_coef = data['entropy_coef']
        else:
            self.init_epsilon = data['init_epsilon']
            self.final_epsilon = data['final_epsilon']
            self.random_steps_fraction = data['random_steps_fraction']
            self.random_steps = int(self.random_steps_fraction * self.total_steps)
            self.target_update = data['target_update']
            self.target_update_every = data['target_update_every']



        envs_names = ['spread', 'coin_gather', 'gathering', 'lbf']
        env_name_index = data['env_name_index']
        self.N_agents = data['N_agents']
        self.env_args = data['env_args']
        self.eval_episodes = data['eval_episodes']
        self.hidden_layers = data['hidden_layers']
        self.cnn = data.get('cnn')
        self.preprocess_action = lambda act: act
        if env_name_index < 2:
            if env_name_index == 1:
                self.env = lambda **args: coin_gather_v0.parallel_env(**args)
            else:
                self.env = lambda **args: simple_spread_v2.parallel_env(**args)
                #self.env_args = {'local_ratio':0, 'N':self.N_agents}
            self.rendered_env = lambda **args: self.env(**dict({'render_mode':'human'},**args))
            self.get_agents_names = lambda env: env.agents
            self.get_obs_space = lambda env, names: {name: env.observation_space(name).shape for name in names}
            self.get_act_space = lambda env, names: {name: env.action_space(name).n for name in names}
            self.step = lambda res: res
        else:
            if env_name_index == 2:
                self.env = lambda **args: gym.make(
                    'gym_cpr_grid:CPRGridEnv-v0', 
                        **args
                    )
                self.get_act_space = lambda env, names: {name: env.action_space.n for name in names}
                self.get_obs_space = lambda env, names: {name: env.observation_space.shape for name in names}
                # self.env_args = {'n_agents':self.N_agents, 'grid_width':25, 'grid_height':7, 'fov_squares_front':5, 'fov_squares_side':2,
                #     #'global_obs':True,
                #     'initial_resource_probability':0.2}
                # self.hidden_layers = [6]
                # self.cnn = [(3, 6, 3)]
            else:
                self.max_food = data['max_food']
                self.env = lambda **args: gym.make(
                    f'Foraging-12x12-{self.N_agents}p-{self.max_food}f-v2', 
                        **args
                    )      
                self.get_act_space = lambda env, names: {name: env.action_space[name].n for name in names}
                self.get_obs_space = lambda env, names: {name: env.observation_space[name].shape for name in names}
                self.preprocess_action = lambda act: act.values()
            self.rendered_env = self.env
            self.get_agents_names = lambda env: [i for i in range(self.N_agents)]
            self.step = lambda res: (res[0], res[1], res[2], [1 if res[3].get("TimeLimit.truncated", False) else 0
                    for _ in range(len(res[1]))], res[3])

        self.render_last_episode = data['render_last_episode']

        
        self.reward_hidden_layers = data['reward_hidden_layers']
        self.N_predictors = data['N_predictors']
        self.perturb_prob = data['perturb_prob']
        self.reward_lr = data['reward_lr']

        self.trajectory_length = data['trajectory_length']
        self.pairs_count = data['pairs_count']
        self.trajectory_buffer_size = self.replay_size // self.trajectory_length // 10
        self.similarity = data['similarity']
        self.reward_weighting = data['reward_weighting']

        self.save_dir = f"saved_models/{envs_names[env_name_index]}_{self.N_agents}_{self.controller}_{self.agent}"
        if self.true_reward:
            self.save_dir += '_true_reward'
        if self.use_full_obs:
            self.save_dir += '_full_obs'
        if self.reward_sharing:
            self.save_dir += f'_{self.perturb_prob}_{self.reward_weighting}'