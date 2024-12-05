from LLM import ChatGPTFunction
from utils import load_file_from_cwd,save_file_to_cwd

class Insighter:
    def __init__(self,*args,**kwargs):        
        self.reset(*args,**kwargs)
        
    def reset(self,env_info=None):
        self.priors = None
        self.insight = None       
        self.config = load_file_from_cwd('config.json')
        self.llm = ChatGPTFunction(**self.config)
        self.env_info = env_info
        
        self.default_item_one = "what aspect of the history and environment is related to the knowlegde? what's the relationship between the knowlegde and the history and environment?"
        self.default_item_two = "why some case in the history is successful and some is not? what action make the case successful? what action make the case fail? What relationship between the action and environment' response have? what's is underlying mechanism behind the relationship? And how to use it?"
        self.default_item_three = "- what's the insight you get? \n- What action you think should take next time?\n- why you think the action is helpful?\n- relationship between the action and environment' response"
        
        
    def generate_insight(self,history,insight_items):
        # Some logic to generate insight
        insight_template = load_file_from_cwd("inference_principle.txt")
        insight_template.replace('[history]',history)
        insight_template.replace('[environment]',self.env_info)
        insight_template.replace('[knowledge]',self.priors)
    
        insight_template.replace('[item_one]',insight_items.get('item_one',self.default_item_one))
        insight_template.replace('[item_two]',insight_items.get('item_two',self.default_item_two))
        insight_template.replace('[item_three]',insight_items.get('item_three',self.default_item_three))
        
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
        return self.priors
    
    def set_knowledge(self,priors):
        self.priors = priors
        
    def save_knowledge(self):
        save_file_to_cwd(self.priors,'priors.txt')
    
