from LLM import ChatGPTFunction
from utils import load_file_from_cwd,save_file_to_cwd

class Insighter:
    def __init__(self,*args,**kwargs):        
        self.reset(*args,**kwargs)
        
    def reset(self,env_info=None):
        self.AK = None
        self.insight = None       
        self.config = load_file_from_cwd('config.json')
        self.llm = ChatGPTFunction(**self.config)
        self.env_info = env_info
        
    def generate_insight(self,history):
        # Some logic to generate insight
        insight_template = load_file_from_cwd("inference_principle.txt")
        insight_template.replace('[history]',history)
        insight_template.replace('[environment]',self.env_info)
        insight_template.replace('[knowledge]',self.AK)
        messages = [
            {
                'role': 'system',
                'content': insight_template
            }
        ]
        self.llm.change_messages(messages)
        response,_ = self.llm.parse([],0)
        self.insight = response
        return self.insight
    
    def get_knowledge(self):
        return self.AK
    
    def set_knowledge(self,AK):
        self.AK = AK
        
    def save_knowledge(self):
        save_file_to_cwd(self.AK,'AK.txt')
    
