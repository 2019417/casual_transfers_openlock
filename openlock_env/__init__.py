from gymnasium.envs.registration import register

register(
    id="openlock_env/OpenlockEnv-v0",
    entry_point="openlock_env.envs:OpenlockEnv"
)