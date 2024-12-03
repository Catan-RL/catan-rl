from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

# from catan_env.catan_env import PettingZooCatanEnv
from catan import RLlib_Catan

# ENV_REGISTRY["catan"] = PettingZooCatanEnv
ENV_REGISTRY["catan"] = RLlib_Catan

env = marl.make_env(environment_name="catanEnv", map_name="catan_scenario")
algo = marl.algos.mappo(hyperparam_source="test")

model = marl.build_model(
    env,
    algo,
    {"core_arch": "mlp", "encode_layer": "8-8", "num_outputs": 176},
)
algo.fit(
    env,
    model,
    stop={"training_iteration": 1},
    local_mode=True,
    num_gpus=0,
    num_workers=2,
    share_policy="group",
    checkpoint_end=False,
)
