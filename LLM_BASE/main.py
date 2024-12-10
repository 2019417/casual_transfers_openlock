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
    message = str(message)
    return YELLOW + message + RESET

def red(message):
    message = str(message)
    return RED + message + RESET

def purple(message):
    message = str(message)
    return PURPLE + message + RESET

def green(message):
    message = str(message)
    return GREEN + message + RESET

def white(message):
    message = str(message)
    return WHITE + message + RESET

def color(message,color_id):
    message = str(message)
    return "\033[1;"+str(color_id)+";40m"+message+RESET

class Runner:
    def __init__(self,pattern,env_seed,env_max_step):
        env_info = load_file_from_cwd('env_info.txt')
        self.env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern=pattern,seed=1,max_step=5)
        self.actioner = Actioner(env_info)
        self.insighter = Insighter(env_info)
        self.learner = Learner(env_info)
    
    def play(self): 
        obs,info = self.env.reset()
        terminated = False
        truncated = False
        solution = info['solution']
        num_of_solution = len(solution)
        finished_count = 0
        print(purple("Welcome to the OpenLockEnv!"))        
    
        while finished_count < num_of_solution:            
            action = self.infence_action(obs=obs)
            action_sequence = []
            print(red("{0:-^50}".format("Environment start")))
            while not (terminated or truncated):
                print_obs(obs)
                print("{0:-^50}".format("Insight"))
                print(yellow(self.actioner.get_insight()))
                print("{0:-^50}".format("Action"))
                print(green(action))
                action = int(input("Your action: "))
                # can alse input insight and knowledge
                obs, reward, terminated, truncated, info = self.env.step(action)
                action = self.infence_action(obs = obs,rewards = reward,terminated = terminated,truncated = truncated)
                action_sequence.append(str(action))
                if terminated:
                    seq = "".join(action_sequence)
                    if seq in solution:                        
                        print("Congratulations! You have finished the task. The solution is: ",seq)
                        finished_count += 1
                        solution.remove(seq)                        
            self.beysian_update()
            print(yellow("beysian update done!"))
            print(white("{:-^50}").format("knowledge updated!"))
            print(white(self.insighter.priors))
            print(red("{0:-^50}".format("reset environment")))
            obs,info = self.env.reset()
            truncated ,terminated ,reward = False,False,0
        # show something, history, insighter
        
        
    def infence_action(self,insight = None,knowledge = None,obs = None,rewards = None,terminated = None,truncated = None):
        # env_response: (obs,rewards,terminated,truncated) (from env.step()) or None
        if knowledge is not None:
            self.insighter.set_knowledge(knowledge)            
        if insight is None:
            history = self.actioner.get_history()            
            insight = self.insighter.generate_insight(history)
        self.actioner.set_insight(insight)
        action = self.actioner.action(obs,rewards,terminated,truncated)
        return action
    
    def beysian_update(self):
        print()
        # update insigher level
        
        # from actioner
        history = self.actioner.get_history()
        insight = self.actioner.get_insight()
        success_try = self.actioner.get_success_try()
        
        # from insighter
        inference_template = self.insighter.inference_principle_template
        inference_item = self.insighter.inference_principle_item
        priors = self.insighter.priors
        
        new_inference_item = self.learner.update_insighter(history,insight,inference_template,inference_item,priors,success_try)  
        self.insighter.inference_principle_item = new_inference_item
        
        # update prior level
        new_priors = self.learner.update_prior(history,success_try)
        self.insighter.priors = new_priors
        save_file_to_cwd(new_priors,'priors.txt')
        
# 不支持render
# pattern: CC3,CC4,CE3,CE4
# max_steps: default 3, 包括开门最多动3次
# size: default 7 ,7 个 lever ,一闪门
# seed: default 0
def print_obs(obs):
    l = len(obs)
    names = [f'levers_{i}' for i in range(l-1)]
    names = names + ['door']
    colors = [obs[i].get('color','no color') for i in range(l)]
    states = [obs[i].get('state','no state') for i in range(l)]
    
    format_names =  [f'{name:^10}' for name in names]
    format_colors = [f'{color:^10}' for color in colors]
    format_states = [f'{state:^10}' for state in states]
    
    name_line = '|'+'|'.join(format_names)+'|'
    color_line = '|'+'|'.join(format_colors)+'|'
    state_line = '|'+'|'.join(format_states)+'|'
    
    print('\n'.join([name_line,color_line,state_line]))
    
    
from tenacity import retry,stop_after_attempt,wait_random
@retry(wait=wait_random(0.2,1),stop=stop_after_attempt(4),reraise=True)
def random_error():
    import random
    print("Random error test")
    if random.random() > 0.2:
        raise Exception("Random error")
    else:
        print("Success")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='play',help='function of the program, train, test ,play')
    
    runner = Runner('CC3',1,5)
    runner.play()
    
    # random_error()
    # print(__name__)


