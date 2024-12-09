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

def print_obs(obs):
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
    
    print('\n'.join([name_line,color_line,state_line]))
    

for e in range(episode):
    print("------" + str(e)+"------")
    observation, info = env.reset()
    solution = info['solution']  # the solution of the environment
    print(solution)
    episode_over = False
    i = 0
    while(not episode_over):
        # action = env.action_space.sample() # agent policy that uses the observation and info
        action = int(input("Your action: "))
        print(f"action_num_{i}:",action)
        i+=1
        observation, reward, terminated, truncated, info = env.step(action) #
        print_obs(observation)
        # print("obs:",observation,"reward:",reward,"terminated:",terminated, "truncated:",truncated)
        episode_over = terminated or truncated
        print(episode_over)
        if episode_over:
            break

env.close()