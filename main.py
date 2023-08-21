from config import *
import numpy as np
from Controllers import get_controller
from typing import Dict, Callable
import time
from matplotlib import pyplot as plt
import gym
import time
from pathlib import Path
import argparse
from eval import evaluate_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("controller_config")
    parser.add_argument("env_config")
    args = parser.parse_args()
    config = get_config(args.controller_config, args.env_config)
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    env = config.env(**config.env_args)
    eval_env = config.env(**config.env_args)
    obs = env.reset()
    rng = np.random.default_rng()
    controller_init = get_controller(config.controller)

    agents_names = config.get_agents_names(env)
    finished = np.zeros(len(agents_names))
    sample_id = 0
    controller = controller_init(len(agents_names), agents_names,
                                 config.get_obs_space(env, agents_names),
                                 config.get_act_space(env, agents_names), config, rng=rng)

    start_time = time.time()
    for global_step in range(config.total_steps+1):
        controller.update_epsilon(global_step)

        if (global_step > config.update_after) and (global_step % config.train_every) == 0:
            controller.train(config.train_batches, step=global_step)

        actions = controller.get_action(obs)
        next_obs, rewards, terminations, infos = env.step(config.preprocess_action(actions))
        for i, name in enumerate(agents_names):
            finished[i] = terminations[name]
        controller.insert_experience(
            obs, actions, next_obs, rewards, terminations, sample_id)
        obs = next_obs
        sample_id = (sample_id + 1) % config.trajectory_database

        if(np.all(finished) or (global_step+1 % config.train_every == 0 and global_step > 0 )):
            controller.finish_path(obs, ((global_step > config.update_after)
                                          and (global_step % config.train_every) == 0)
                                         or infos.get('TimeLimit.truncated', False))
        if np.all(finished):
            obs = env.reset()

        if(global_step % config.steps_per_epoch == 0):
            print(f"Step {global_step}/{config.total_steps}")
            print(evaluate_env(controller, config, config.eval_episodes, eval_env))
            controller.save_models(
                config.save_dir, global_step//config.steps_per_epoch)
            print(
                f"Elapsed time = {time.time() - start_time:.2f} seconds", flush=True)
            start_time = time.time()
    env.close()
    eval_env.close()
