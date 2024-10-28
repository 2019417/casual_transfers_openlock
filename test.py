import gymnasium as gym
import openlockenv #不加会报错

# 不支持render
# pattern: CC3,CC4,CE3,CE4
# max_steps: default 3, 包括开门最多动3次
# size: default 6 ,6 个 lever ,一闪门
# seed: default 0
env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern='CC3',seed=1)
#observation, info = env.reset()
#solution = info['solution'] # the solution of the environment
#print(solution)
episode = 10
max_step = 5
for e in range(episode):
    print("------" + str(e)+"------")
    observation, info = env.reset()
    solution = info['solution']  # the solution of the environment
    print(solution)
    episode_over = False
    for _ in range(max_step):
        action = env.action_space.sample() # agent policy that uses the observation and info
        print("action:",action)
        observation, reward, terminated, truncated, info = env.step(action) #
        print("obs:",observation,"reward:",reward,"terminated:",terminated)
        episode_over = terminated or truncated
        print(episode_over)
        if episode_over:
            break


env.close()