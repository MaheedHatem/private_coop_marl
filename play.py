from config import *
import numpy as np
import time

def get_action(obs, agents_names):
    actions = {name: 0 for name in agents_names}
    for name in agents_names:
        act = input(f"agent {name} ")
        if act == 'w':
            act = 2
        elif act == 'a':
            act = 1
        elif act == 's':
            act = 3
        elif act == 'd':
            act = 0
        elif act == '4' or act == '5':
            act = int(act)
        else:
            raise Exception(f"Unsupported Action {act}")
        actions[name] = act
    return actions

if __name__ == '__main__':
    config = Config()
    env = config.env(**config.env_args)
    obs = env.reset()
    env.render()
    agents_names = config.get_agents_names(env)
    finished = np.zeros(len(agents_names))

    done = False
    while not done:
        actions = get_action(obs, agents_names)
        obs, rewards, terminations, truncations, infos = config.step(env.step(actions))
        for i, name in enumerate(agents_names):
            finished[i] = terminations[name] or truncations[name]
        done = np.all(finished)
        env.render()
        time.sleep(0.1)
    env.close()