import gymnasium as gym
import openlockenv
import random as rd
import os


def test_CC3():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3')
    while env.unwrapped != env:
        env = env.unwrapped
    solutions = env.get_solution()
    i, j = rd.sample(range(2), 2)
    assert solutions[i][0] == solutions[j][0]
    for solution in solutions:
        for a in solution:
            a = int(a)
            o, r, end, t, i = env.step(a)
            assert r == 0
            assert not end
            assert t == False
        o, r, end, t, i = env.step(7)
        assert r == 1
        assert end == True
        env.reset()

    env.reset()
    for i in range(3):
        env.step(0)

    o, r, end, t, i = env.step(7)
    assert t == True


def test_CC4():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC4')
    while env.unwrapped != env:
        env = env.unwrapped
    solutions = env.get_solution()
    i, j = rd.sample(range(3), 2)
    assert solutions[i][0] == solutions[j][0]
    for solution in solutions:
        for a in solution:
            a = int(a)
            o, r, end, t, i = env.step(a)
            assert r == 0
            assert not end
            assert t == False
        o, r, end, t, i = env.step(7)
        assert r == 1
        assert end == True
        env.reset()

    env.reset()
    for i in range(3):
        env.step(0)

    o, r, end, t, i = env.step(7)
    assert t == True


def test_open_door_directly():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3')
    _,info = env.reset()
    o ,r,end, t ,i = env.step(7)
    assert not end
    assert not t

def test_CE3():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CE3')
    while env.unwrapped != env:
        env = env.unwrapped
    solutions = env.get_solution()
    i, j = rd.sample(range(2), 2)
    assert solutions[i][1] == solutions[j][1]
    for solution in solutions:
        for a in solution:
            a = int(a)
            o, r, end, t, i = env.step(a)
            assert r == 0
            assert not end
            assert t == False
        o, r, end, t, i = env.step(7)
        assert r == 1
        assert end == True
        env.reset()

    env.reset()
    for i in range(3):
        env.step(0)

    o, r, end, t, i = env.step(7)
    assert t == True


def test_CE4():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CE4')
    while env.unwrapped != env:
        env = env.unwrapped
    solutions = env.get_solution()
    i, j = rd.sample(range(3), 2)
    assert solutions[i][1] == solutions[j][1]
    for solution in solutions:
        for a in solution:
            a = int(a)
            o, r, end, t, i = env.step(a)
            assert r == 0
            assert not end
            assert t == False
        o, r, end, t, i = env.step(7)
        assert r == 1
        assert end == True
        env.reset()

    env.reset()
    for i in range(3):
        env.step(0)

    o, r, end, t, i = env.step(7)
    assert t == True


def test_reset():
    env1 = gym.make(id='openlockenv/OpenlockEnv-v0', seed=1)
    while env1.unwrapped != env1:
        env1 = env1.unwrapped
    
    solutions1 = env1.get_solution()
    
    env1.reset()
    assert solutions1 == env1.get_solution()
    env2 = gym.make(id='openlockenv/OpenlockEnv-v0', seed=1)
    while env2.unwrapped != env2:
        env2 = env2.unwrapped
    assert solutions1 == env2.get_solution()


def test_wrong_pattern():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC5')
    while env.unwrapped != env:
        env = env.unwrapped
    assert env._OpenlockEnv__pattern == 'CC3'


def test_get_solution():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3')
    _,info = env.reset()
    solution = info['solution']
    assert solution[0][0] == solution[1][0]
    while env.unwrapped != env:
        env = env.unwrapped
    env.get_solution()

    
def test_observeration():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3')
    o,info = env.reset()
    solutions = info['solution']
    solution = set(''.join(solutions))
    levers = o[0:-1]
    door = o[-1]
    assert all(
                lever['color'] == 'grey' if str(i) in solution else 'white' and 'state' in lever and lever['state'] == 0
                for i,lever in enumerate(levers)
               ) and 'door' in door and 'state' in door and door['state'] == 0

def test_random_action_CC3():
    max_step = 5
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3',seed = 1,max_step = max_step)
    o,info = env.reset()
    solutions = info['solution']
    attemps = 15    
    for at in range(attemps):
        env.reset()
        trace = '' 
        end = False
        while(not end):
            action = env.action_space.sample()
            trace += str(action)
            o,r,ter,tru,i = env.step(action)
            
            is_ter = any(solution in trace for solution in solutions) and action == 7
            is_tru = len(trace) > max_step
            try:
                assert is_tru == tru
                assert (is_ter == ter) if not tru else True
            except AssertionError:
                while env.unwrapped != env:
                    env = env.unwrapped
                print("trace:",trace,"env.trace:",env._OpenlockEnv__trace)
                print("solution:",solutions,"env.solution:",env._OpenlockEnv__solutions)
                print("is_ter:",is_ter,"ter:",ter)
                print("is_tru:",is_tru,"tru:",tru)
                raise AssertionError
            end = ter or tru
        
    
    

def test_start_script():
    import gymnasium as gym
    import openlockenv

    # 不支持render
    # 不支持render
    # pattern: CC3,CC4,CE3,CE4
    # max_steps: default 3, 包括开门最多动3次
    # size: default 6 ,6 个 lever ,一闪门
    # seed: default 0
    env = gym.make(id='openlockenv/OpenlockEnv-v0',pattern='CC3',seed=1,max_step = 5)
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
        
        
def test_push_able():
    env = gym.make(id='openlockenv/OpenlockEnv-v0', pattern='CC3')
    _,info = env.reset()
    solutions = info['solution']
    # print(solutions)
    first_levers = set([int(solution[0]) for solution in solutions])
    for _ in range(20):
        l = env.action_space.sample()
        if l not in first_levers:
            obs, *_ = env.step(l)
            # print(l)
            assert obs[l]['state'] == 0
        else:
            obs, *_ = env.step(l)
            assert obs[l]['state'] == 1
            break