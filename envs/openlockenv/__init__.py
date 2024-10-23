from gymnasium.envs.registration import register

register(
    id="openlockenv/OpenlockEnv-v0",
    entry_point="openlockenv.envs:OpenlockEnv",
    order_enforce=False
)