from catan_env.game.game import Game
from catan_env.catan_env import PettingZooCatanEnv
from catan_agent.models.build_agent_model import build_agent_model
from catan_agent.ppo.ppo import PPO
from catan_agent.ppo.arguments import get_args

if __name__ == "__main__":
    env = PettingZooCatanEnv()
    central_policy = build_agent_model(device="cpu")

    args = get_args()

    agent = PPO(central_policy, args)
    
    entropy_coef = args.entropy_coef_start

    agent.entropy_coef

