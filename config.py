import lbforaging
import gym
import coin_gathering.coin_gathering
from misc import TimeLimit

import yaml
import json

def get_config(controller_config_file, env_config_file):
    with open(f"Configs/controllers/{controller_config_file}.yaml", "r") as f:
        controller_data = yaml.safe_load(f)
    with open(f"Configs/envs/{env_config_file}.yaml", "r") as f:
        env_data = yaml.safe_load(f)
    return Config(controller_data, env_data)
        
    

class Config:
    def __init__(self, controller_data, env_data):
        self.controller = controller_data['controller']
        self.agent = controller_data['agent']
        self.reward_sharing = self.agent == 'DQNRewardAgent' or self.agent =='ACRewardAgent'
        self.true_reward = controller_data['true_reward']
        if self.true_reward:
            assert self.controller == 'DecentralizedController' and self.agent == 'DQNAgent'
        if self.reward_sharing:
            assert self.controller == 'DecentralizedController'
        #self.use_state = False
        self.use_full_obs = controller_data['use_full_obs']
        self.device = controller_data['device']

        self.replay_size = controller_data['replay_size'] #100000 works for spread
        self.gamma = controller_data['gamma']
        self.lr = controller_data['lr'] #worked with 0.0001 too i think
        self.train_every = controller_data['train_every']
        self.train_batches = controller_data['train_batches']
        self.update_after = controller_data['update_after']
        self.batch_size = controller_data['batch_size']
        self.total_steps = env_data['total_steps']
        self.steps_per_epoch = env_data['steps_per_epoch']
        self.update_predictor_steps = controller_data.get('update_predictor_steps', self.total_steps)
        if self.agent == 'ACAgent' or self.agent == 'ACRewardAgent':
            self.value_coef = controller_data['value_coef']
            self.entropy_coef = controller_data['entropy_coef']
        else:
            self.init_epsilon = controller_data['init_epsilon']
            self.final_epsilon = controller_data['final_epsilon']
            self.random_steps_fraction = controller_data['random_steps_fraction']
            self.random_steps = int(self.random_steps_fraction * self.total_steps)
            self.target_update = controller_data['target_update']
            self.target_update_every = controller_data['target_update_every']


        self.hidden_layers = controller_data['hidden_layers']
        self.cnn = controller_data.get('cnn')

        envs_names = ['gathering', 'lbf', 'CoinGathering']
        env_name_index = env_data['env_name_index']
        self.N_agents = env_data['N_agents']
        self.env_args = env_data['env_args']
        self.max_episode_steps = self.env_args['max_episode_steps']
        self.preprocess_action = lambda act: act
        if env_name_index == 0:
            self.env = lambda **args: TimeLimit(gym.make(
                'gym_cpr_grid:CPRGridEnv-v0', 
                    **args
                ), self.max_episode_steps)
            self.get_act_space = lambda env, names: {name: env.action_space.n for name in names}
            self.get_obs_space = lambda env, names: {name: env.observation_space.shape for name in names}
            # self.env_args = {'n_agents':self.N_agents, 'grid_width':25, 'grid_height':7, 'fov_squares_front':5, 'fov_squares_side':2,
            #     #'global_obs':True,
            #     'initial_resource_probability':0.2}
            # self.hidden_layers = [6]
            # self.cnn = [(3, 6, 3)]
        elif env_name_index == 1:
            self.max_food = env_data['max_food']
            coop = env_data.get('coop', '')
            grid_size = env_data['grid_size']
            self.env = lambda **args: TimeLimit(gym.make(
                f'Foraging-{grid_size}x{grid_size}-{self.N_agents}p-{self.max_food}f{coop}-v2', 
                    **args
                ), self.max_episode_steps)
            self.get_act_space = lambda env, names: {name: env.action_space[name].n for name in names}
            self.get_obs_space = lambda env, names: {name: env.observation_space[name].shape for name in names}
            self.preprocess_action = lambda act: act.values()
        else:
            self.env = lambda **args: TimeLimit(gym.make('CoinGathering-v0', **args), self.max_episode_steps)
            self.get_act_space = lambda env, names: {name: env.action_space[name].n for name in names}
            self.get_obs_space = lambda env, names: {name: env.observation_space[name].shape for name in names}
            self.preprocess_action = lambda act: list(act.values())
        self.rendered_env = self.env
        self.get_agents_names = lambda env: [i for i in range(self.N_agents)]
        self.eval_episodes = env_data['eval_episodes']
        self.render_last_episode = env_data['render_last_episode']

        if self.reward_sharing:
            self.trajectory_database = controller_data['trajectory_database']
            self.reward_hidden_layers = controller_data['reward_hidden_layers']
            self.N_predictors = controller_data['N_predictors']
            self.perturb_prob = controller_data['perturb_prob']
            self.reward_lr = controller_data['reward_lr']

            self.trajectory_length = controller_data['trajectory_length']
            self.pairs_count = controller_data['pairs_count']
            self.trajectory_buffer_size = self.trajectory_database // self.trajectory_length
            self.similarity = controller_data['similarity']
            self.reward_weighting = controller_data['reward_weighting']
        else:
            self.trajectory_database = self.replay_size

        self.save_dir = f"saved_models/{envs_names[env_name_index]}_{self.N_agents}_{self.controller}_{self.agent}"
        if self.true_reward:
            self.save_dir += '_true_reward'
        if self.use_full_obs:
            self.save_dir += '_full_obs'
        if self.reward_sharing:
            self.save_dir += f'_{self.perturb_prob}_{self.reward_weighting}'