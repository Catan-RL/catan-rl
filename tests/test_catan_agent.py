from catan_agent.models.action_head import ActionHead
# from catan_agent.models.action_heads_module import ActionHead
from catan_agent.distributions import Categorical
import torch

if __name__ == "__main__":
    actions = ActionHead(947, 10)

    actions.forward(torch.rand(947), torch.tensor([1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float))