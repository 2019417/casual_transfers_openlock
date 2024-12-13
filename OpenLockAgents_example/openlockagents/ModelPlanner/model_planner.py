import torch
import os
import numpy as np

from openlockagents.common.agent import Agent
from openlockagents.common.common import DEBUGGING

'''from openlockagents.A2C_modelbased.model import DiscretePolicy
from openlockagents.A2C_modelbased.model import Value'''
from openlockagents.ModelPlanner.model import World
from openlockagents.ModelPlanner.core import update_params
from openlockagents.ModelPlanner.utils.replay_memory import Memory


class ModelPlanner(Agent):
    def __init__(self, env, state_size, action_size, params, require_log=True):
        """
        Init A2C agent for OpenLock env

        :param env
        :param state_size
        :param action_size
        :param params
        """
        super(ModelPlanner, self).__init__("A2C", params, env)
        if require_log:
            super(ModelPlanner, self).setup_subject(human=False, project_src=params["src_dir"])

        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.attempt_rewards = []
        self.trial_rewards = []
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.trial_percent_solution_found = []
        self.trial_percent_attempt_success = []
        self.trial_length = []
        self.total_loss = [0]
        self.states_loss = [0]
        self.rewards_loss = [0]
        self.memory = Memory()
        
        self.gamma = params["gamma"]
        self.lr = params["learning_rate"]
        self.epsilon = params["epsilon"]
        self.l2_reg = params["l2_reg"]
        self.batch_size = params["batch_size"]
        self.use_gpu = params["use_gpu"] and torch.cuda.is_available()
        self.gpuid = params["gpuid"]
        self.tensor = torch.cuda.DoubleTensor if self.use_gpu else torch.DoubleTensor
        self.action_tensor = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor
        self.render = params["use_physics"]
        self.simulated_sequences = {}  #{key = (0, 0, 14): item = reward0 + rewar1 + reward2}
        #self.simulated_sequences_second = {} #{key = 0-14: item = reward1 + reward2}
        #self.simulated_memory = Memory() #存储模拟的sequences用于更新world


        self.STEPS = 0

        self.env.lever_index_mode = "role"

        self._build_model()
        self._build_sequences()

    def _build_sequences(self):
        for i in range(0,self.action_size-1):
            #self.simulated_sequences_second[(i, )] =0
            for j in range(0, self.action_size-1):
                self.simulated_sequences[(i, j ,14)] = 0
                

    def _build_model(self):
        self.world = World(self.action_size, self.state_size)
        if self.use_gpu:
            torch.cuda.set_device(self.gpuid)
            self.world = self.world.cuda()

        self.optimizer_world = torch.optim.Adam(self.world.parameters(), lr = self.lr)

    def planning(self,initial_state):
        
        for k in self.simulated_sequences.keys():
            #self.simulated_memory.push()
            state = self.tensor(initial_state.astype(np.float32)).unsqueeze(0)
            self.simulated_sequences[k] = 0
            for i, action in enumerate(k):
                next_state, reward = self.world(self.action_tensor(np.array([action]).astype(np.float32)).unsqueeze(0), state)
                #self.simulated_memory.push(state, action, 1 if i<2 else 0, next_state, reward)
                reward = reward.detach()
                state = next_state
                self.simulated_sequences[k] += self.gamma**i * reward
        #print(self.simulated_sequences)
                
    def run_trial_planner(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        trial_count,
        iter_num,
        testing=False,
        specified_trial=None,
        fig=None,
        fig_update_rate=100,
    ):
        """
        Run a computer trial using A2C.

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param testing:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial
        )

        # print('Scenario name: {}, iter num: {}, trial count: {}, trial name: {}'.format(scenario_name, iter_num, trial_count, trial_selected))

        trial_reward = 0
        self.env.attempt_count = 0
        attempt_reward = 0
        reward = 0
        attempt_success_count = 0
        #TODO 对simulated进行根据model进行赋值
        state = self.env.reset()
        self.planning(state)
        sorted_sequences = {k: v for k, v in sorted(self.simulated_sequences.items(), key=lambda item: item[1], reverse = True)}
        #print(sorted_sequences)
        for k in sorted_sequences.keys():
            if self.determine_trial_finished(attempt_limit):
                continue
            else:
                done = False
                state = self.env.reset()
                step = 0
                while not done:
                    prev_attempt_reward = attempt_reward
                    prev_reward = reward
                    action_idx = k[step]
                    step +=1
                # convert idx to Action object (idx -> str -> Action)
                    action = self.env.action_map[self.env.action_space[action_idx]]
                    next_state, reward, done, opt = self.env.step(action)

                    mask = 0 if done else 1
                    self.memory.push(state, action_idx, mask, next_state, reward)

                    if self.render:
                        self.env.render()
                    trial_reward += reward
                    attempt_reward += reward
                    state = next_state

            self.finish_attempt()

            if DEBUGGING:
                pass
                # self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, 1.0)
                # print(self.logger.cur_trial.attempt_seq[-1].action_seq)

            assert (
                self.env.cur_trial.cur_attempt.cur_action is None
                and len(self.env.cur_trial.cur_attempt.action_seq) == 0
            )
            assert attempt_reward == self.env.cur_trial.attempt_seq[-1].reward

            self.attempt_rewards.append(attempt_reward)

            attempt_reward = 0
            self.total_attempt_count += 1

            if opt["attempt_success"]:
                attempt_success_count += 1

            # if fig is not None and self.env.attempt_count % fig_update_rate == 0:
            #     fig = self.log_values([self.average_trial_rewards, self.attempt_rewards],
            #                           fig,
            #                           ['Average Trial Reward', 'Attempt Reward'])

        print(
            "Trial end, avg_reward:{}, solutions found:{}/{}".format(
                trial_reward / attempt_limit,
                len(self.env.get_completed_solutions()),
                len(self.env.get_solutions()),
            )
        )
        self.env.cur_trial.trial_reward = trial_reward
        self.trial_rewards.append(trial_reward)
        self.average_trial_rewards.append(trial_reward / self.env.attempt_count)
        self.trial_switch_points.append(len(self.attempt_rewards))
        self.trial_percent_solution_found.append(
            len(self.env.get_completed_solutions()) / len(self.env.get_solutions())
        )
        self.trial_percent_attempt_success.append(
            attempt_success_count / self.env.attempt_count
        )
        self.trial_length.append(self.env.attempt_count)

        self.finish_trial(trial_selected, test_trial=testing)


    def update(self, batch, i_iter, scenario, i_trial):
        """
        Update the actor-critic model with A2C
        Args:
        """
        total_loss, rewards_loss, states_loss = update_params(
            self.world,
            self.optimizer_world,
            batch,
            self.tensor,
            self.action_tensor,
            self.epsilon, #the coeffecient of reward loss
            self.l2_reg,
            i_iter,
            scenario,
            i_trial,
        )
        self.total_loss.append(total_loss.detach().numpy())
        self.rewards_loss.append(rewards_loss.detach().numpy())
        self.states_loss.append(states_loss.detach().numpy())
        print(self.total_loss)


    def save(self, path, iter_num):
        fd = "/".join(path.split("/")[:-1])
        os.makedirs(fd, exist_ok=True)
        torch.save(
            self.policy.state_dict(),
            os.path.join(path, "{:06d}.policy".format(iter_num)),
        )
        torch.save(
            self.value.state_dict(), os.path.join(path, "{:06d}.value".format(iter_num))
        )

    def load(self, path):
        self.world.load_state_dict(torch.load(path + ".world"))


    def finish_subject(self, strategy="Modelplanner", transfer_strategy="Modelplanner"):
        super(ModelPlanner, self).finish_subject(strategy, transfer_strategy, self)
