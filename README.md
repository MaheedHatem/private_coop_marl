# Privacy-Preserving Decentralized Actor-Critic for Cooperative Multi-Agent Reinforcement Learning

This is the official code implementation for the paper "Privacy-Preserving Decentralized Actor-Critic for Cooperative Multi-Agent Reinforcement Learning". It also includes implementation for independent actor-critic for both the coin-gathering and level-based-foraging environments.

## Setup
In order to run the files install python>=3.9 then run:

`pip install -r requirements.txt `

## Training

To train, run the command:

`python main.py train <controller_config> <environment_config> <output_directory> [-s seed] [-n parallel_envs]`

The <controller_config> parameter is one of:
- ac_reward
- ac_reward_coin
- ac_dec
    
ac_reward is the config file for running PPDAC, and ac_reward_coin is the version for the coin2 environment. ac_dec is the config file for running IAC. The corresponding config file is found under Configs/controllers. To run PPDAC with perturbation, change the value of perturb_prob in the ac_reward.yaml or ac_reward_coin.yaml files.

The <environment_config> parameter is one of:
    - coin2
    - coin3
    - lbf2
    - lbf3
    - lbf3_4
    - lbf2_coop
    
each corresponding to one of the experiments in the paper.

## Evaluation
To evaluate a run, run the command:

`python main.py eval <controller_config> <environment_config> <output_directory> [-s seed]`

## Comparisonc
To plot a comparison between multiple runs, create a .yaml file <filename.yaml> under Comparisons folder with the same format as other files in the directory then run the following command.

`python main.py compare <filename>`

Note that comparisons require that a full training run and an evaluation run to guarantee the required files are created in the output folder.