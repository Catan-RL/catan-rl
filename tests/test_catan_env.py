import numpy as np
from pettingzoo.test import api_test
from pettingzoo.utils import AssertOutOfBoundsWrapper, OrderEnforcingWrapper

from catan_env.catan_env import PettingZooCatanEnv

if __name__ == "__main__":
    np.random.seed(0)

    env = PettingZooCatanEnv()
    wrapped = AssertOutOfBoundsWrapper(env)
    wrapped = OrderEnforcingWrapper(wrapped)
    api_test(env, num_cycles=10, verbose_progress=True)
