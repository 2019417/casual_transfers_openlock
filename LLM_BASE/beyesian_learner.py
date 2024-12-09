from LLM import ChatGPTFunction
from utils import load_file_from_cwd,save_file_to_cwd
import json


class Learner:
    def __init__(self,env_info,**kwargs):
        self.reset(env_info)
    
    def reset(self,env_info):
        config = json.loads(load_file_from_cwd("config.json"))
        self.env_info = env_info
        self.llm = ChatGPTFunction(**config)
        self.insighter_likehood = load_file_from_cwd("insighter_likehood.txt")
        self.prior_likehood = load_file_from_cwd("prior_likehood.txt")
    
    def update_insighter(self,history,insights,inference_template,inference_item,prior,success_try):
        insighter_likehood = self.insighter_likehood
        insighter_likehood = insighter_likehood.replace("[history]",history)
        insighter_likehood = insighter_likehood.replace("[inference_principle]",inference_template)
        insighter_likehood = insighter_likehood.replace("[advice_principle]",inference_item)
        insighter_likehood = insighter_likehood.replace("[environment]",self.env_info)
        insighter_likehood = insighter_likehood.replace("[prior]",prior)
        insighter_likehood = insighter_likehood.replace("[insight]",insights)
        insighter_likehood = insighter_likehood.replace("[success_try]",success_try)
        messages = [
            {
                'role': 'system',
                'content': insighter_likehood
            }
        ] 
        self.llm.change_messages(messages)
        response,_ = self.llm.parse([],0)    
        return response
    
    def update_prior(self,history,success_try):
        prior_likehood = self.prior_likehood
        prior_likehood = prior_likehood.replace("[history]",history)
        prior_likehood = prior_likehood.replace("[environment]",self.env_info)
        prior_likehood = prior_likehood.replace("[success_try]",success_try)
        messages = [
            {
                'role': 'system',
                'content': prior_likehood
            }
        ]
        self.llm.change_messages(messages)
        response,_ = self.llm.parse([],0)
        return response
       
       
       
    