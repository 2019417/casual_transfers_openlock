import logging.config
import queue
from LLM import ChatGPTFunction
import json
from utils import load_file_from_cwd
import logging

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('actioner')


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
        logger.info("Generate new action sequence")
        
        action_template = load_file_from_cwd('action_principle.txt')
        action_template = action_template.replace("[insight]",self.insight)
        action_template = action_template.replace("[environment]",self.env_info)
        action_template = action_template.replace("[solutions]",json.dumps(self.known_solutions))
        
        logger.debug(f"Action template: {action_template}")
        

        for i in range(10):
            try:
                messages = [{
                'role': 'system',
                'content': action_template
                }]
                self.llm.change_messages(messages) 
                response, *_ = self.llm.parse([],0)
                action_list = json.loads(response['content'])
                action_list.sort(key=lambda x: x['step'],reverse=True)
                self.action_seq = [x['action'] for x in action_list]
                break
            except Exception as e:
                print(f"action sequence error {i} time")
                print("try again......")
                messages[0]['content'] += "\nMust generate action sequence format!!"
                if i == 3:
                    logger.error(f"Action sequence error: {response}")
                    raise e
        
        
        
        logger.debug(f"Action sequence: {self.action_seq}")

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
            logger.info("Task terminated or truncated")
            logger.info(f"Rewards: {rewards}")
            logger.info(f"Now Action sequence: {self.now_attempt}")
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
            #TODO better expression ?
            round_res = str(self.history.get())
            res += f"Round {i}: {round_res}\n"
        return res
    
    def get_success_try(self):
        res = ""
        for i, solution in enumerate(self.known_solutions):
            res += f"Solution {i}: {solution}\n"
        return res
        
    def get_insight(self):
        return self.insight
        
    def set_insight(self,insight):
        self.insight = insight
        
    
if __name__ == "__main__":
    actioner = Actioner()

    