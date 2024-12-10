import logging.config
import queue
from LLM import ChatGPTFunction
import json
from utils import load_file_from_cwd,NpEncoder
import logging

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('actioner')

class Actioner:
    def __init__(self, env_info = None):
        self.reset(env_info) 

    def reset(self,env_info):
        self.history = History(2)
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
        action_template = action_template.replace("[solutions]",json.dumps(self.history.get_successful_round(5),indent=2,cls=NpEncoder))
        
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
                logger.debug(f"Action list: {action_list}")
                action_list.sort(key=lambda x: x['step'])
                self.action_seq = [x['action'] for x in action_list]
                break
            except Exception as e:
                print(f"action sequence error {i} time")
                logger.error(f"Action sequence error {i} time")
                print("try again......")
                messages[0]['content'] += "\nMust generate action sequence format!!"
                if i == 3:
                    logger.error(f"Action sequence error: {response}")
                    raise e
                
        logger.debug(f"Action sequence: {self.action_seq}")

    def __attempt__(self,step,obs,action,insight,round):
        return {
            'round': round,
            'step': step,
            'obs': obs,
            'action':action,
            "insight":insight
        }

    def action(self,obs, rewards=None,terminated=None,truncated=None,round = None,action=None,**kwargs):
        if(len(self.action_seq)==0):
            self.__generate_action_seq() 
        # check input 
        # TODO: check ?
        if(terminated or truncated):
            logger.info("Task terminated or truncated")
            logger.info(f"Rewards: {rewards}")
            logger.info(f"Now Action sequence: {self.now_attempt}")
            self.history.append_attemp(self.now_attempt,terminated)            
            if(rewards>0):
                self.known_solutions.append(self.action_seq)
            self.reset_action_seq()
            return -1
        else:
            if action is None:
                action = self.action_seq.pop(0)
            self.__step += 1
            self.now_attempt.append(self.__attempt__(self.__step,obs,action,self.insight,round))
            return action
    
    def get_history(self):
        return self.history
            
    def set_known_solutions(self,known_solutions):
        self.known_solutions = known_solutions
        
    def append_known_solutions(self,known_solution):
        self.known_solutions.append(known_solution)
        
    def get_success_try(self):
        return self.history.get_successful_round(5)
        
    def get_insight(self):
        return self.insight
        
    def set_insight(self,insight):
        self.insight = insight
                
class History:
    def __init__(self,size):
        self.size = size
        self.success_history = [None for i in range(size)]
        self.si = 0
        self.failed_history = [None for i in range(size)]
        self.fi = 0
        self.current_round = []
    
    def append_round(self,round,succeed):
        if succeed is True:
            self.success_history[self.si] = round
            self.si += 1
            self.si = self.si % self.size
        elif succeed is False:
            self.failed_history[self.fi] = round
            self.fi += 1
            self.fi = self.fi % self.size
    
    def append_attemp(self,attempt,succeed=None):
        self.current_round.append(attempt)
        if succeed is True:
            self.success_history[self.si] = self.current_round
            self.si += 1
            self.si = self.si % self.size            
            self.current_round = []
        elif succeed is False:
            self.failed_history[self.fi] = self.current_round
            self.fi += 1
            self.fi = self.fi % self.size
            self.current_round = []
        
    def get_successful_round(self,k):
        return list(self.success_history)[-k:]
    
    def get_failed_round(self,k):
        return list(self.failed_history.queue)[-k:]
        
    @staticmethod
    def formate_rounds(cls,rounds):
        if rounds is None or len(rounds) == 0:
            return ""
        rounds = list(rounds)        
        res = ""
        for round in rounds:
            if round is None:
                continue
            id = round[0]['round']
            res += f"Round {id}: \n"
            for attempt in round:
                res += f"Step {attempt['step']}: \n"
                res += f"Action: {attempt['action']}\n"
                res += f"Observation: \n"
                res += cls.format_obs(cls,attempt['obs'])
        return res
    
    @staticmethod
    def format_obs(cls,obs):
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
        
        res = '\n'.join([name_line,color_line,state_line])
        res += '\n'
        return res

    def to_json(self,indent=2):
        return json.dumps({
            'success_history': (self.success_history),
            'failed_history': (self.failed_history)
        },indent=indent,cls=NpEncoder)
    
    def __str__(self):
        res = "Successful rounds: \n"
        res += History.formate_rounds(self,self.success_history)
        res += "Failed rounds: \n"
        res += History.formate_rounds(self,self.failed_history)
        return res
    
if __name__ == "__main__":
    his = History(5)
    his.append_attemp({'round':1,'step':1,'obs':[{'color':'red','state':'up'},{'color':'blue','state':'down'}],'action':1},True)
    his.append_attemp({'round':2,'step':2,'obs':[{'color':'red','state':'up'},{'color':'blue','state':'down'}],'action':2},False)
    his.append_attemp({'round':3,'step':3,'obs':[{'color':'red','state':'up'},{'color':'blue','state':'down'}],'action':3})
    his.append_attemp({'round':3,'step':4,'obs':[{'color':'red','state':'up'},{'color':'blue','state':'down'}],'action':4})
    his.append_attemp({'round':3,'step':5,'obs':[{'color':'red','state':'up'},{'color':'blue','state':'down'}],'action':5},True)
    print(his.to_json(indent=2))
        
