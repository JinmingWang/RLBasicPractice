from typing import *
import random
import torch
import torch.nn as nn
from Environment.State import State
from Environment.Action import Action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个类型表示多个torch.Tensor
TensorTuple = Tuple[torch.Tensor, ...]


class MemoryTuple:
    def __init__(self, state: State, action: Action, reward: float, next_state: State, terminate_flag: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminate_flag = terminate_flag


    def __iter__(self):
        # 因为要使用zip，所以必须实现__iter__方法
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.terminate_flag

    @staticmethod
    def makeBatch(batch_data: List['MemoryTuple']) -> TensorTuple:
        states, actions, rewards, next_states, terminate_flags = zip(*batch_data)
        return State.makeBatch(states), \
            Action.makeBatch(actions), \
            torch.tensor(rewards, dtype=torch.float32, device=device), \
            State.makeBatch(next_states), \
            torch.tensor(terminate_flags, dtype=torch.float32, device=device)