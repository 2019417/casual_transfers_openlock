# 六个核桃大作业

🐮🐮🐮🐮🐮🐮


## 1.环境配置 :wrench:  

先安装包
```bash
pip install -r requirements.txt
pip install openlockenv-0.0.2-py3-none-any.whl
```


```python

import gymnasium as gym
import openlockenv #不加会报错

# 不支持render
# pattern: CC3,CC4,CE3,CE4 
# max_steps: default 3, 包括开门最多动3次
# size: default 6 ,6 个 lever ,一闪门
# seed: default 0
env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern='CC3')
observation, info = env.reset()
solution = info['solution'] # the solution of the environment
print(solution)
episode = 10
for e in range(episode):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action) # 
    episode_over = terminated or truncated

env.close()
```