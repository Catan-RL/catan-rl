from pettingzoo.test import api_test

from catan_env.catan_env import PettingZooCatanEnv

if __name__ == "__main__":
    env = PettingZooCatanEnv()
    api_test(env, num_cycles=10, verbose_progress=True)
