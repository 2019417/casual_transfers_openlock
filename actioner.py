class Actioner:
    def __init__(self, action):
        self.history = {
            'privous_attempt': [],
            
            "now_attempt":[
                # {
                #     "obs": "start_obs",
                #     'action': "start_action"
                # },
                # {
                #     'obs': 'next_obs',
                #     'action': "next_action"
                # }
            ]
        }
        self.action = action

    def __update_attempt(self,action,obs):
        self.history['now_attempt'].append(
            {
                'obs':obs,
                'action':action
            }
        )
    
    def __update_history(self):
        self.history['privous_attempt'].append(self.history['now_attempt'])
        self.history['now_attempt'] = []

    def __gpt_format(self):
        pass
    
    def __action_reduction(self,action):
        pass
    
    def get_action(self,obs,insight):
        pass
    
    def update(self,obs,rewrads,terminated,truncated,info,insight):
        pass
    
    def get_insight(self,history):
        if history:
            return "This is an CC environment"
        else:
            return "This is an CC environment"
    def set_insight(self,insight):
        self.insight = insight
    
    
    