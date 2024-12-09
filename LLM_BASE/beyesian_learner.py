from LLM import ChatGPTFunction
from utils import load_file_from_cwd,save_file_to_cwd

class Learner:
    def __init__(self,env_info,**kwargs):
        self.reset(env_info)
    
    def reset(self,env_info):
        config = load_file_from_cwd("config.json")
        self.env_info = env_info
        self.llm = ChatGPTFunction(**config)
    
    def update_insighter(self,history,insights,inference_template,inference_item):
        return "not complete"
    
    def update_prior(self,insights,inference_princple,priors,history):
        return "not complete"
       
       
       
    