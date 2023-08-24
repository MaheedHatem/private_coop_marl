from config import *
import numpy as np
from Controllers import get_controller
from Controllers import BaseController
from typing import Dict, Callable
import time
import torch
import random
from matplotlib import pyplot as plt
from logger import init_logging
import gym
import time
import argparse
import logging

def evaluate_env(controller: BaseController, config: Config, episodes: int, env) -> Dict[str, np.ndarray]:
    agents_names = controller.names
    finished = np.zeros(controller.n_agents)
    rews = {name: np.zeros(episodes) for name in agents_names}
    act_dim = config.get_act_space(env, agents_names)
    act_count = {name: {i: 0 for i in range(act_dim[name])} for name in agents_names}
    rews['total'] = np.zeros(episodes)
    rews['min'] = np.zeros(episodes)
    rews['max'] = np.zeros(episodes)
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_rewards = {name: [] for name in agents_names}
        while not done:
            actions = controller.get_action((np.expand_dims(obs,1)), True)
            obs, rewards, terminations, infos = env.step(np.squeeze(actions))
            for name in agents_names:
                episode_rewards[name].append(rewards[name])
            for i, name in enumerate(agents_names):
                finished[i] = terminations[name]
            done = np.all(finished)
        for name in agents_names:
            agent_reward = np.array(episode_rewards[name]).sum()
            rews['total'][episode] += agent_reward
            rews[name][episode] = agent_reward
        rews['min'][episode] = np.min(np.array([rews[name][episode] for name in agents_names]))
        rews['max'][episode] = np.max(np.array([rews[name][episode] for name in agents_names]))
    return {name: rews[name].mean() for name in rews.keys()}

def evaluate_run(args):
    if args.seed != None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    config = get_config(args.controller_config, args.env_config, 1, args.directory, args.seed)
    init_logging(f"{config.save_dir}/eval.log")
    logging.info(args)
    env = config.eval_env(**config.env_args)
    env.reset()
    rng = np.random.default_rng()
    controller_init = get_controller(config.controller)

    agents_names = config.get_agents_names(env)
    controller = controller_init(len(agents_names), agents_names, 
        config.get_obs_space(env, agents_names), 
        config.get_act_space(env, agents_names), config, rng=rng)
    total_steps = config.total_steps//config.steps_per_epoch
    scores = np.zeros((total_steps,3 + len(agents_names)))
    model_dir = args.directory
    for index in range(total_steps):
        controller.load_models(model_dir, index)
        res = evaluate_env(controller, config, 100, env)
        scores[index,0] = res['total']
        scores[index,1] = res['min']
        scores[index,2] = res['max']
        for i, name in enumerate(agents_names):
            scores[index, i+3] = res[name]
        logging.info(f"{index}: {res}")
    env.close()
    steps = np.arange(total_steps) * config.steps_per_epoch
    plt.plot(np.arange(total_steps) * config.steps_per_epoch, scores)
    plt.show()
    np.savetxt(f"{model_dir}/results.csv", np.column_stack((steps, scores)), delimiter=',')
