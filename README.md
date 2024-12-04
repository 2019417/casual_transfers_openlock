# 四个核桃大作业

🐮🐮🐮🐮🐮🐮


## 1. 环境配置 :wrench:  

先安装包
```bash
pip install -r requirements.txt
pip install openlockenv-0.0.4-py3-none-any.whl
```

卸载包
```bash
pip uninstall openlockenv
```

```python
import gymnasium as gym
import openlockenv #不加会报错

# 不支持render
# pattern: CC3,CC4,CE3,CE4
# max_steps: default 3, 包括开门最多动3次
# size: default 7 ,7 个 lever ,一闪门
# seed: default 0
max_step = 5
env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern='CC3',seed=1,max_step=max_step)
#observation, info = env.reset()
#solution = info['solution'] # the solution of the environment
#print(solution)
episode = 10
for e in range(episode):
    print("------" + str(e)+"------")
    observation, info = env.reset()
    solution = info['solution']  # the solution of the environment
    print(solution)
    episode_over = False
    i = 0
    while(not episode_over):
        action = env.action_space.sample() # agent policy that uses the observation and info
        print(f"action_num_{i}:",action)
        i+=1
        observation, reward, terminated, truncated, info = env.step(action) #
        print("obs:",observation,"reward:",reward,"terminated:",terminated, "truncated:",truncated)
        episode_over = terminated or truncated
        print(episode_over)
        if episode_over:
            break

env.close()
```
## 2. OpenLock环境
![](https://camo.githubusercontent.com/9e206156f9295b836f2e661d471d76f36828d6d00a111d1d34a7b11c0b0a495b/687474703a2f2f7777772e6d6a65646d6f6e64732e636f6d2f70726f6a656374732f4f70656e4c6f636b2f436f6753636931385f6f70656e6c6f636b5f736f6c7574696f6e732e676966)

a. 包含7个杆和一个门。杆的颜色分为浅灰色和深灰色，浅灰色的杆永远锁。需要按一定顺序推深灰色的杆以解锁门，然后才能推开门。其中颜色是能被观察到的。所以原文提到的abstract-level是这个解锁逻辑，而instance-level是颜色位置属性的对应关系。obs(16维) = {7*(status_lever) + 7*(color_lever)+door_open+door_lock}。

b. 需要找到所有的解决方法才能逃离房间。新的房间只有杆的位置变了，相应杆的属性和门解锁的逻辑没有改变。一次验证agent的causal learning能力。

c. 一次attempt只有3steps，包含开门。以上图为例，$L_0$在$L_1$或$L_2$推开之前应该是被锁住，推不动的？原文所举的例子推上杆导致下杆被推开不知道是不是笔误？所以个最佳的agent在一个房间最多需要N+1个attempts（N个解锁方法）。第一次挨个推深灰色的杆确定其在图里的第一层还是第二层，然后后面N次直接找到N个解锁方法。
![b](figs/1.png)

## 3. 实验方法

### a. Training
训练阶段只包含3个深灰色杆的场景。每完成一个trial（找到所有解锁方法或者attempt的次数用完），换到下一个位置不一样但解锁逻辑相同的房间进行下一个trial。
### b. Test
#### congruent cases
CE3-CE4 CC3-CC4 (train on 3,test on 4)
#### incongruent cases
CC3-CE4 CE3-CC4 (train on 3,test on 4)
#### baseline
CC4-CC4 CE4-CE4 (train on 4,test on 4)

### c. Experimental Setup
##### transfer-based agent（本文提出的）
(1) a training trial 最多30 attempts。每次trial的房间即杆的位置都不同，相同的房间只看一次。

(2) 测试时对每个房间也只有1次机会，30个attempt。
##### Model-free RL agent
(1) a training trial 最多700 attempts。可以重复地看相同的房间（200次）。

(2) 测试时对每个房间也有200次机会，700个attempt。什么是all six 3-lever and all five 4-lever？

(3) reward: 对每个解决方法只在第一次找到时才给个奖励。
![2](figs/2.png)
