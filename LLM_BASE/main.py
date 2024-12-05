import gymnasium as gym
import openlockenv #不加会报错

# 不支持render
# pattern: CC3,CC4,CE3,CE4
# max_steps: default 3, 包括开门最多动3次
# size: default 7 ,7 个 lever ,一闪门

# seed: default 0

env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern='CC3',seed=1,max_step=5)
env.reset()
env.close()

def train():
    pass


def test():
    pass