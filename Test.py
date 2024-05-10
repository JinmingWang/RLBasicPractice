import cv2

from Environment.GameEnv import GameEnv
from Agents.A2CAgent import A2CAgent
from TrainUtils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

"""
Loading model from Runs/20230822_070250/Model_it_5500
Episode 0 reward: 38.00, survive step: 1680
Episode 1 reward: 15.00, survive step: 152
Episode 2 reward: 6.00, survive step: 55
Episode 3 reward: 63.00, survive step: 1501
Episode 4 reward: 3.00, survive step: 28
Episode 5 reward: 83.00, survive step: 10580
Episode 6 reward: 49.00, survive step: 6412
Episode 7 reward: 14.00, survive step: 181
Episode 8 reward: 2.00, survive step: 82
Episode 9 reward: 36.00, survive step: 5126
"""

def test():
    train_config, env_config, agent_config = loadTrainConfig("TrainConfig.yaml")
    agent_config["epsilon"] = 0.0
    env = GameEnv(env_config)
    agent = A2CAgent(agent_config)

    for i in range(100):
        # 与环境互动获得数据
        agent.eval()
        env.reset()
        is_terminate = False
        while not is_terminate:
            state = env.current_state
            action = agent.getAction(state)
            next_state, reward, is_terminate = env.step(action)

            # 可视化
            env.render(f"r={env.total_reward:.2f}")
            env.current_state.render()
            cv2.waitKey(1)

        print(f"Episode {i} reward: {env.total_reward:.2f}, survive step: {env.t}")


if __name__ == '__main__':
    test()