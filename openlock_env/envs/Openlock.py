import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import random as rd


class OpenlockEnv(gym.Env):
    metadata = {'render.modes': ['None']}

    def __init__(self, size=6, pattern='CC3', max_step=3, seed=0):
        ''''
            size: number of lever >= 6
            pattern: the pattern of solutions: CC3 CE3 CC4 CE4    

            OpenLock Environment
        '''
        super().__init__()
        self.__size = size if size >= 6 else 6
        self.__pattern = pattern
        if(not self.is_good_pattern(pattern)):
            print("wrong pattern string ,using default pattern CC3")
            self.__pattern = 'CC3'
            
        self.__seed = seed
        self.__max_steps = max_step if max_step >= 3 else 3
        self.__step = 0
        self.__trace = ""
        
        self.__state = [0 for e in range(size+1)]
        self.__solutions = self.__sample_solution(self.__pattern)
        self.action_space = spaces.Discrete(size+1)
        self.observation_space = spaces.MultiBinary(size+1)
    
    
    def step(self, action: int):
        if (action == self.__size and self.__check_door()):
            self.__state[action] = 1
        else:
            self.__state[action] = 1

        # add step
        self.__step += 1
        # add trace
        self.__trace = self.__trace + str(action)

        return self.__get_obs(), self.__get_reward(action), self.__truncated(), self.__terminated(), self.__get_info()

    def __sample_solution(self, pattern):
        rd.seed(self.__seed)
        if ('CC' in pattern):
            if ('3' in pattern):
                c1, c2, c3 = rd.sample(range(1, self.__size), 3)
                return [f"{c1}{c2}", f"{c1}{c3}"]
            else:
                c1, c2, c3, c4 = rd.sample(range(1, self.__size), 4)
                return [f"{c1}{c2}", f"{c1}{c3}", f"{c1}{c4}"]
        elif ("CE" in pattern):
            if ('3' in pattern):
                c1, c2, c3 = rd.sample(range(1, self.__size), 3)
                return [f"{c2}{c1}",f"{c3}{c1}"]
            else:
                c1, c2, c3, c4 = rd.sample(range(1, self.__size), 4)
                return [f"{c2}{c1}",f"{c3}{c1}", f"{c4}{c1}"]
        else:
            print("wrong pattern string ,using default pattern CC")

    def is_good_pattern(self, pattern):
        if (pattern not in ['CC3', 'CE3', 'CC4', 'CE4']):
            return False
        else:
            return True

    # unfinished
    def __get_info(self):
        return 'unfinished info'

    def __check_door(self):
        for solution in self.__solutions:
            if (solution in self.__trace):
                return True
        return False

    def __get_obs(self):
        return np.array(self.__state)

    def __get_reward(self, action):
        if (action == self.__size and self.__check_door()):
            return 1
        else:
            return 0

    def __truncated(self):
        if (self.__step > self.__max_steps):
            return True
        else:
            return False

    def __terminated(self):
        if (self.__state[self.__size] == 1):
            return True
        else: 
            return False
                
    def reset(self, **karg):
        self.__seed = karg['seed'] if 'seed' in karg else self.__seed
        self.__step = 0
        self.__trace = ''
        self.__state = [0 for e in range(self.__size+1)]
        self.__solutions = self.__sample_solution(self.__pattern)

    def render(self):
        # TODO
        return super().render()

    def close(self):
        # TODO
        return super().close()
