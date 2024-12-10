# TA版ENV

🐮🐮🐮🐮🐮🐮


# 1. Environment
## git clone two repo
## for gl, GUL of pyglet
```bash
apt-get install libglu1-mesa libglu1-mesa-dev 
apt install libgl1-mesa-glx
```
## for constraint
```bash
conda install conda-forge::python-constraint
```

## for box2d
```bash
conda install conda-forge::pybox2d
```

## for pygraphviz
```bash
conda install --channel conda-forge pygraphviz
```

## if error:pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
```bash
xvfb-run -s "-screen 0 640x480x24" python *.py
```



# 2. code change

```bash
#OpenLockAgents_example/openlockagents/common/common.py
ROOT_DIR = "/home/wajuda/Proj/OpenLockAgents_example"  #change to yours to save log
```
```bash
#OpenLockAgents_example/openlockagents/A2C/a2c_open_lock.py
#TA   外循环是房间设置，内循环是次数
for trial_num in range(0, len(possible_trials)):
        for iter_num in tqdm(range(params["num_training_iters"])):

#changed by wangjunda  内循环是房间设置，外循环是次数。似乎更符合原文设置
for iter_num in tqdm(range(params["num_training_iters"])):
    for trial_num in range(0, len(possible_trials)):
```

```bash
# not use log
# OpenLockAgents_example/openlockagents/common/agent.py
#我注释掉了几处log的部分 比如： 有需要可以更改
'''self.logger = SubjectLogger(
            subject_id=self.subject_id,
            participant_id=participant_id,
            age=age,
            gender=gender,
            handedness=handedness,
            eyewear=eyewear,
            major=major,
            start_time=time.time(),
            random_seed=random_seed,
        )''' #stop log
```

# 3. experiment setting
```bash
#OpenLockAgents_example/openlockagents/A2C/a2c_open_lock.py
#TA是主要在原来的main()基础上改成了a2c_main_like_human(） 3为iter，可以按论文设为200，700是max_attempt
a2c_main_like_human('CC3-CC4', 0, 'negative_immovable_unique_solutions', 3, 700, None)

#原论文的main()
params["num_training_trials"] = params["train_num_trials"]
#TA实现的a2c_main_like_human(）
possible_trials = agent.get_random_order_of_possible_trials(
        params["train_scenario_name"]
    )
```

```text
可以按照类似的修改方式按照一个标准测试几个model-free RL
现在的设置是
train（cc3 200iter 6rooms 700attempt）
transfer（cc4 200iter 1rooms 700attempt）
即transfer只有一个房间，按原论文是要循环5个4-lever房间，可以改，但要统一
```
