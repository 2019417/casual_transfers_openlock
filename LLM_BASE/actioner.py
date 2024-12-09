import logging.config
import queue
from LLM import ChatGPTFunction
import json
from utils import load_file_from_cwd
import logging


class Actioner:
    def __init__(self, env_info = None):
        self.reset(env_info) 

    def reset(self,env_info):
        self.history = queue.Queue(20)
        self.now_attempt = [] 
        self.action_seq = []
        self.insight = "No insight available"
        # TODO add environment info
        self.env_info = env_info
        self.config = json.loads(load_file_from_cwd('config.json'))
        self.known_solutions = []
        self.llm = ChatGPTFunction(**self.config)
        self.__step = 0
    
    def reset_action_seq(self):
        self.action_seq = []
        self.__step = 0
        self.now_attempt = []

    def __generate_action_seq(self):
        action_template = load_file_from_cwd('action_principle.txt')
        action_template = action_template.replace("[insight]",self.insight)
        action_template = action_template.replace("[environment]",self.env_info)
        action_template = action_template.replace("[solutions]",json.dumps(self.known_solutions))
        messages = [{
            'role': 'system',
            'content': action_template
        }]
        self.llm.change_messages(messages) 
        response, *_ = self.llm.parse([],0)
        action_list = json.loads(response['content'])
        action_list.sort(key=lambda x: x['step'],reverse=True)
        self.action_seq = [x['action'] for x in action_list]        

    def __attempt__(self,step,obs,action,insight):
        return {
            'step': step,
            'obs': obs,
            'action':action,
            "insight":insight
        }

    def action(self,obs, rewards=None,terminated=None,truncated=None,**kwargs):
        if(len(self.action_seq)==0):
            self.__generate_action_seq() 
        # check input 
        # TODO: check ?
        if(terminated or truncated):
            self.history.put(self.now_attempt)
            if(rewards>0):
                self.known_solutions.append(self.action_seq)
            self.reset_action_seq()
            return -1
        else:
            action = self.action_seq.pop(0)
            self.__step += 1
            self.now_attempt.append(self.__attempt__(obs,self.__step,action,self.insight))
            return action

    
    def get_history(self):
        res = "The history of this task is: \n"
        for i in range(self.history.qsize()):
            res += f"Round {i}: {json.dumps(self.history.get(),indent=2)}\n"
        return res
    
    def get_success_try(self):
        return self.known_solutions
        
    def get_insight(self):
        return self.insight
        
    def set_insight(self,insight):
        self.insight = insight
    
    
if __name__ == "__main__":
    actioner = Actioner()

    