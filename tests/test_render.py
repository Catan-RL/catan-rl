import numpy as np

from catan_env.catan_env import PettingZooCatanEnv

# from pettingzoo.utils import AssertOutOfBoundsWrapper, OrderEnforcingWrapper


if __name__ == "__main__":
    np.random.seed(0)

    env = PettingZooCatanEnv(interactive=False)
    # wrapped = AssertOutOfBoundsWrapper(env)
    # wrapped = OrderEnforcingWrapper(wrapped)

    human_agent_idx = 0

    num_episodes = 1
    for episode in range(num_episodes):
        env.reset()

        human_agent = env.agents[human_agent_idx]

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            print(f"Reward: {reward}")

            if termination or truncation:
                action = None
            else:

                # if agent == human_agent:
                #     # action = session.query(...)
                # else:
                # our model/agent goes here:
                #     # action = model.predict(observation)

                if observation is None:
                    print("Observation is None")
                    action = env.action_space(agent).sample()
                else:
                    action = env.action_space(agent).sample(
                        observation["action_mask"]
                    )
            env.step(action)

        env.close()

        print(f"Episode {episode + 1} ended")
