import os
import sys
import json
import pprint
import gym

# project root dir, three directories up
from shutil import copytree, ignore_patterns
'''print('hahahhah')
print(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'''

ROOT_DIR = "/home/wajuda/Proj/OpenLockAgents_example"


DEBUGGING = True if "pydevd" in sys.modules else False


def load_json_config(path):
    with open(path) as json_data_file:
        config_data = json.load(json_data_file)
        return config_data


def write_source_code(project_src_path, destination_path):
    copytree(
        project_src_path,
        destination_path,
        ignore=ignore_patterns(
            "*.mp4", "*.pyc", ".git", ".gitignore", ".gitmodules"
        ),
    )


def print_message(trial_count, attempt_count, message, print_message=True):
    if print_message:
        print("T{}.A{}: ".format(trial_count, attempt_count) + message)


def pre_env_instantiation_setup(params, bypass_confirmation=False):
    print("PARAMETERS:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(params)

    if not bypass_confirmation:
        input("Press Enter to confirm...")

    # copy the source code to the tmp directory - we need to save a copy when we start, not when we create an agent. Source code could change in between time program is launched and agents are created (especially if running multiple agents)

    # allows disabling writing the source code (used for replay)
    if params["src_dir"] is not None:
        write_source_code(ROOT_DIR, params["src_dir"])

    env = make_env(params)
    return env


def make_env(params):
    # setup initial env
    env = gym.make("openlock-v1")
    env.use_physics = params["use_physics"]
    env.initialize_for_scenario(params["train_scenario_name"])
    if "effect_probabilities" in params.keys():
        env.set_effect_probabilities(params["effect_probabilities"])
    return env
