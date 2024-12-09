import gymnasium as gym
import openlockenv #不加会报错
import argparse

from actioner import Actioner
from insighter import Insighter

from beyesian_learner import Learner
from utils import load_file_from_cwd,save_file_to_cwd

RED = '\033[1;31;40m'
GREEN = '\033[1;32;40m'
WHITE = '\033[1;37;40m'
YELLOW = '\033[1;33;40m'
PURPLE = '\033[1;35;40m'
RESET = '\033[0m'

def yellow(message):
    return YELLOW + message + RESET

def red(message):
    return RED + message + RESET

def purple(message):
    return PURPLE + message + RESET

def green(message):
    return GREEN + message + RESET

def white(message):
    return WHITE + message + RESET



def color(message,color_id):
    return "\033[1;"+str(color_id)+";40m"+message+RESET


class Runner:
    def __init__(self,pattern,env_seed,env_max_step):
        env_info = load_file_from_cwd('env_info.txt')
        self.env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern=pattern,seed=1,max_step=5)
        self.actioner = Actioner(env_info)
        self.insighter = Insighter(env_info)
        self.learner = Learner(env_info)
    
    def play(self): 
        obs = self.env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.infence_action()
            # chose your action 
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.beysian_update()
        
        # show something, history, insighter
        
        
    def infence_action(self,insight = None,knowledge = None, env_response = None):
        # env_response: (obs,rewards,terminated,truncated) (from env.step()) or None
        if knowledge is None:
            self.insighter.set_knowledge(knowledge)            
        if insight is None:
            history = self.actioner.get_history()            
            insight = self.insighter.generate_insight(history)            
        self.actioner.set_insight(insight)
        if env_response is None:
            action = self.actioner.action()
        else:
            action = self.actioner.action(env_response[0],env_response[1],env_response[2],env_response[3])
        return action
    
    def beysian_update(self):
        print()
        # update insigher level
        
        # from actioner
        history = self.actioner.get_history()
        insight = self.actioner.get_insight()
        
        # from insighter
        inference_template = self.insighter.inference_principle_template
        inference_item = self.insighter.inference_principle_item
        
        new_inference_item = self.learner.update_insighter(history,insight,inference_template,inference_item)        
        self.insighter.inference_principle_item = new_inference_item
        
        # update prior level
        new_priors = self.learner.update_prior(insight,self.insighter.inference_principle,self.insighter.priors,history = history)
        self.insighter.priors = new_priors
        save_file_to_cwd(new_priors,'priors.txt')
        
        
# 不支持render
# pattern: CC3,CC4,CE3,CE4
# max_steps: default 3, 包括开门最多动3次
# size: default 7 ,7 个 lever ,一闪门
# seed: default 0
def print_obs(obs):
    l = len(obs)
    names = ['levers_'+i for i in range(l-1)]
    names = names + ['door']
    colors = [obs[i].get('color','no color') for i in range(l)]
    states = [obs[i].get('state','no state') for i in range(l)]
    
    format_names = [f'{name:^10}' for name in names]
    format_colors = [f'{color:^10}' for color in colors]
    format_states = [f'{state:^10}' for state in states]
    
    name_line = '|'+'|'.join(names)+'|'
    color_line = '|'+'|'.join(format_names)+'|'
    state_line = '|'+'|'.join(format_states)+'|'
    
    print('\n'.join([name_line,color_line,state_line]))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='play',help='function of the program, train, test ,play')
        

