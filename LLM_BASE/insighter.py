import logging.config
from LLM import ChatGPTFunction
from utils import load_file_from_cwd,save_file_to_cwd
import json
import logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('insighter')

class Insighter:
    def __init__(self,*args,**kwargs):        
        self.reset(*args,**kwargs)
        
    def reset(self,env_info=None):
        self.priors = ""
        self.insight = ""
        self.config = json.loads(load_file_from_cwd('config.json'))
        self.llm = ChatGPTFunction(**self.config)
        self.env_info = env_info
        
        self.inference_principle_template = load_file_from_cwd("inference_principle.txt")
        self.inference_principle_item = "- what's the insight you get? \n- What action you think should take next time?\n- why you think the action is helpful?\n- relationship between the action and environment' response"
        self.inference_principle = self.build_inferece_principle("")
       
    def build_inferece_principle(self,history):
        inference_template = self.inference_principle_template
        # inference context
        inference_template = inference_template.replace('[history]',history)
        # learning context
        inference_template = inference_template.replace('[item]',self.inference_principle_item)
        inference_template = inference_template.replace('[environment]',self.env_info)
        inference_template = inference_template.replace('[knowledge]',self.priors)
        return inference_template
        
    def generate_insight(self,history):
        logger.info("Generate new insight")
        inference_template = self.build_inferece_principle(history)
        messages = [
            {
                'role': 'system',
                'content': inference_template
            }
        ]
        logger.debug(f"Insight template: {inference_template}")
        self.llm.change_messages(messages)
        response, *_ = self.llm.parse([],0)
        self.insight = response['content']
        logger.debug(f"Insight: {self.insight}")
        return self.insight

    def get_knowledge(self):
        return self.priors

    def get_principle_item(self):
        return self.inference_principle_item

    def set_principle_item(self,inference_principle_item):
        self.inference_principle_item = inference_principle_item
    
    def set_knowledge(self,priors):
        self.priors = priors
        
    
