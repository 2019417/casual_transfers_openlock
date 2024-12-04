import queue
from LLM import ChatGPTFunction
import json
from utils import load_file_from_cwd
import re

class Actioner:
    def __init__(self, env_info = None):
        self.history = queue.Queue(20)
        self.now_attempt = [self.__attempt__(-1,None,None,None)] 
        self.action_seq = []
        self.insight = "No insight available"
        self.env_info = env_info
        self.config = json.loads(load_file_from_cwd('config.json'))
        self.llm = ChatGPTFunction(**self.config)
        
    def reset(self):
        pass

    def __gpt_format(self):
        pass
        
    def __generate_action_seq(self):
        # generate action prompt to gpt
        
        # action_prompt = action_prompt_template + obs + insight
        action_template = load_file_from_cwd('action_principle.txt')
        action_template = action_template.replace("[insight]",self.insight)
        action_template = action_template.replace("[environment]",self.env_info)
        
        self.llm.change_messages(action_template) 
        response, _ = self.llm.parse([],0)
        self.action_seq = json.loads(response)
        
    def __attempt__(self,step,obs,action,insight):
        
        return {
            'step': step,
            'obs': obs,
            'action':action,
            "insight":insight
        }
    
    def action(self,obs,terminated,truncated,**kwargs):
        if(len(self.action_seq)==0):
            self.__generate_action_seq() 
        if(terminated or truncated):
            self.history.put(self.now_attempt)
            return -1
        else:
            action = self.action_seq.pop(0)
            step = self.now_attempt[-1].get('step') + 1
            self.now_attempt.append(self.__attempt__(obs,step,action,self.insight))
            return action
        
    def get_insight(self):
        return self.insight
        
        
    def set_insight(self,insight):
        self.insight = insight
    
    
    