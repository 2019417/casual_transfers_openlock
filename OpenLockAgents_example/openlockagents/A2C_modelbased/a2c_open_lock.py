#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import time
import numpy as np
import sys
import os
import pickle
import atexit
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from openlock.settings_trial import PARAMS, IDX_TO_PARAMS
from openlock.settings_scenario import select_scenario
from openlock.envs.openlock_env import ObservationSpace

from openlockagents.common.common import ROOT_DIR, pre_env_instantiation_setup
from openlockagents.A2C_modelbased.a2c_agent import A2CModelAgent


def create_reward_fig():
    # creating the figure
    # plt.ion()
    fig = plt.figure()
    fig.set_size_inches(20, 5)
    # plt.pause(0.0001)
    return fig


def a2c_model_based(str_scenario: str, gpuid: int, reward_mode: str='negative_immovable_unique_solutions', num_iter: int=100, attempt_limit: int=700,
         load_path: str=None):
    torch.set_default_tensor_type("torch.DoubleTensor")
    # Set up environment
    params = {
        'train_action_limit': 3
    }
    if str_scenario in ['CC4', 'CE4']:
        params['train_scenario_name'] = str_scenario
        transfer_tag = str_scenario
        params['test_scenario_name'] = None
    elif str_scenario in ['CC3-CC4', 'CC3-CE4', 'CE3-CC4', 'CE3-CE4']:
        transfer_tag = str_scenario.replace('-', 'to')
        params['train_scenario_name'] = str_scenario.split('-')[0]
        params['test_scenario_name'] = str_scenario.split('-')[1]
    else:
        assert False, "Check the str_scenario: {}".format(str_scenario)

    params["gamma"] = 0.99
    params["learning_rate"] = 0.001
    params["epsilon"] = 0.95
    params["l2_reg"] = 1e-3
    params["batch_size"] = 2048
    params["use_gpu"] = True
    params["gpuid"] = gpuid
    params["src_dir"] = "/tmp/openlocklearner/" + str(hash(time.time())) + "/src/"
    random_seed = 1234
    params["use_physics"] = False
    params[
        "full_attempt_limit"
    ] = (
        False
    )  # run to the full attempt limit, regardless of whether or not all solutions were found
    params["num_training_iters"] = num_iter
    params["train_attempt_limit"] = attempt_limit
    params["test_attempt_limit"] = attempt_limit
    params["test_action_limit"] = 3


    # RL specific settings
    params["data_dir"] = os.path.dirname(ROOT_DIR) + "/OpenLockRLResults/subjects"

    params["reward_mode"] = reward_mode

    scenario = select_scenario(
        params["train_scenario_name"], use_physics=params["use_physics"]
    )

    env = pre_env_instantiation_setup(params, bypass_confirmation=True)
    env.use_physics = params["use_physics"]
    env.full_attempt_limit = params["full_attempt_limit"]
    # set up observation space
    env.observation_space = ObservationSpace(
        len(scenario.levers), append_solutions_remaining=False
    )
    # set reward mode
    env.reward_mode = params["reward_mode"]
    print("Reward mode: {}".format(env.reward_mode))
    np.random.seed(random_seed)
    env.seed(random_seed)

    # dummy agent
    agent = A2CModelAgent(env, 1, 1, params, require_log=False)
    trial_selected = agent.setup_trial(
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
    )
    env.reset()

    state_size = agent.env.observation_space.multi_discrete.shape[0]
    action_size = len(env.action_space)
    agent = A2CModelAgent(env, state_size, action_size, params)
    save_path = os.path.join(
        params["data_dir"],
        "3rd_model_log/a2c_model_based-{}-{}-{}".format(
            transfer_tag if transfer_tag else params["train_scenario_name"],
            params["reward_mode"],
            agent.subject_id,
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    agent.env.reset()
    if load_path:
        agent.load(load_path)
        print("load model from {}".format(load_path))

    agent.env.human_agent = False
    agent.type_tag = "{}-{}-A2C".format(
        transfer_tag if transfer_tag else params["train_scenario_name"],
        params["reward_mode"],
    )
    possible_trials = agent.get_random_order_of_possible_trials(
        params["train_scenario_name"]
    )
    fig = create_reward_fig()
    # Training
    for iter_num in tqdm(range(params["num_training_iters"])):
        for trial_num in range(0, len(possible_trials)):
            agent.run_trial_a2c(
                scenario_name=params["train_scenario_name"],
                fig=fig,
                action_limit=params["train_action_limit"],
                attempt_limit=params["train_attempt_limit"],
                trial_count=trial_num,
                iter_num=iter_num,
                specified_trial=possible_trials[trial_num],
            )
            fig, data = agent.log_values(
                [
                    agent.trial_length,
                    agent.trial_percent_attempt_success,
                    agent.trial_percent_solution_found,
                    agent.average_trial_rewards,
                    agent.attempt_rewards,
                ],
                fig,
                [
                    "Attempt Count Per Trial",
                    "Percentage of Successful Attempts in Trial",
                    "Percentage of Solutions Found in Trial",
                    "Average Trial Reward",
                    "Attempt Reward",
                ],
                agent.type_tag,
            )
            pickle.dump(
                (agent.type_tag, data, params),
                open(os.path.join(save_path, "log.pkl"), "wb"),
            )
            # update
            print(f'before update:{len(agent.memory)}')
            if len(agent.memory) > agent.batch_size:
                batch = agent.memory.sample()
                print("update with bs:{}".format(len(batch.state)))
                agent.update(batch, iter_num)
                agent.memory.clear()
                print(f'after update:{len(agent.memory)}')

    # Training
    for iter_num in tqdm(range(params["num_training_iters"])):
        for trial_num in range(0, len(possible_trials)):
            agent.run_trial_model(
                scenario_name=params["train_scenario_name"],
                #fig=fig,
                action_limit=params["train_action_limit"],
                attempt_limit=params["train_attempt_limit"],
                trial_count=trial_num,
                iter_num=iter_num,
                specified_trial=possible_trials[trial_num],
            )
            '''pickle.dump(
                (agent.type_tag, data, params),
                open(os.path.join(save_path, "log.pkl"), "wb"),
            )'''
            # update
            
            if len(agent.simulated_memory) > agent.batch_size:
                batch = agent.simulated_memory.sample()
                print("update with bs:{}".format(len(batch.state)))
                agent.update(batch, iter_num)
                agent.simulated_memory.clear()
        #agent.save(save_path, trial_num)
    # Testing
    if params["test_scenario_name"] == "CE4" or params["test_scenario_name"] == "CC4":
        possible_trials = agent.get_random_order_of_possible_trials(
            params["test_scenario_name"]
        )
        print(f'possible_test trials: {possible_trials}')
        testing_trial = possible_trials[0]
        for iter_num in tqdm(range(params["num_training_iters"])):
            agent.run_trial_a2c(
                scenario_name=params["test_scenario_name"],
                fig=fig,
                action_limit=params["test_action_limit"],
                attempt_limit=params["test_attempt_limit"],
                trial_count=0,
                iter_num=iter_num,
                specified_trial=testing_trial,
                testing=True,
            )
            fig, data = agent.log_values(
                [
                    agent.trial_length,
                    agent.trial_percent_attempt_success,
                    agent.trial_percent_solution_found,
                    agent.average_trial_rewards,
                    agent.attempt_rewards,
                ],
                fig,
                [
                    "Attempt Count Per Trial",
                    "Percentage of Successful Attempts in Trial",
                    "Percentage of Solutions Found in Trial",
                    "Average Trial Reward",
                    "Attempt Reward",
                ],
                agent.type_tag,
            )
            pickle.dump(
                (agent.type_tag, data, params),
                open(os.path.join(save_path, "log.pkl"), "wb"),
            )
            # update
            if len(agent.memory) > agent.batch_size:
                batch = agent.memory.sample()
                print("update with bs:{}".format(len(batch.state)))
                agent.update(batch, iter_num)
                agent.memory.clear()
        #agent.save(save_path, iter_num=10)
    '''print(
        "Trial complete for subject {}. Average reward: {}".format(
            agent.logger.subject_id, agent.average_trial_rewards[-1]
        )
    )''' #stop log
    print(
        "Average reward: {}".format(agent.average_trial_rewards[-1]
        )
    )
    fig.savefig(os.path.join(save_path, "log.png"))


if __name__ == "__main__":
    # main()
    a2c_main_like_human('CC3-CC4', 0, 'negative_immovable_unique_solutions', 200, 700, None)
    #main()
