from config import *
import numpy as np
from Controllers import get_controller
import time
from matplotlib import pyplot as plt
import gym
import time
from pathlib import Path
import argparse
import torch
import random
from logger import init_logging
import logging
import sys
from compare import compare_results
from eval import evaluate_env, evaluate_run

def train(args):
    config = get_config(args.controller_config, args.env_config, args.num_parallel, args.directory, args.seed)
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    init_logging(f"{config.save_dir}/train.log")
    logging.info(args)
    env = config.env(**config.env_args)
    eval_env = config.eval_env(**config.env_args)
    obs = env.reset()
    if args.seed != None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    controller_init = get_controller(config.controller)

    agents_names = config.get_agents_names(env)
    sample_id = 0
    controller = controller_init(len(agents_names), agents_names,
                                 config.get_obs_space(env, agents_names),
                                 config.get_act_space(env, agents_names), config, rng=rng)

    start_time = time.time()
    for global_step in range(0,config.total_steps+1, config.num_parallel):
        controller.update_epsilon(global_step)

        if (global_step > config.update_after) and (global_step % config.train_every) == 0:
            controller.train(config.train_batches, step=global_step)

        actions = controller.get_action(obs)
        next_obs, rewards, terminations, infos = env.step(actions)
        obs_to_save = np.copy(next_obs)
        truncated = [info.get('TimeLimit.truncated', False) for info in infos]
        for j in range(config.num_parallel):
            if(terminations[j]):
                obs_to_save[:, j] = infos[j]['terminal_observation']
        controller.insert_experience(
            obs, actions.T, obs_to_save, rewards, terminations, truncated, sample_id)
        obs = next_obs
        sample_id = (sample_id + 1) % config.trajectory_database

        if(global_step % config.steps_per_epoch == 0):
            logging.info(f"Step {global_step}/{config.total_steps}")
            logging.info(evaluate_env(controller, config, config.eval_episodes, eval_env))
            controller.save_models(
                config.save_dir, global_step//config.steps_per_epoch)
            logging.info(
                f"Elapsed time = {time.time() - start_time:.2f} seconds")
            start_time = time.time()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("controller_config")
    train_parser.add_argument("env_config")
    train_parser.add_argument("directory")
    train_parser.add_argument('-n', "--num_parallel", default=4, type=int)
    train_parser.add_argument('-s', "--seed", default=None, type=int)
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument("controller_config")
    eval_parser.add_argument("env_config")
    eval_parser.add_argument("directory")
    eval_parser.add_argument('-s', "--seed", default=None, type=int)
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument("comparison_list")
    args = parser.parse_args()
    if(args.command == "train"):
        train(args)
    elif (args.command == "eval"):
        evaluate_run(args)
    elif (args.command == "compare"):
        compare_results(args)