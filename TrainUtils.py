import yaml
from typing import *
import random
from Agents.AgentUtils import MemoryTuple, TensorTuple


def loadYaml(config_path: str) -> Dict[str, Any]:
    """
    加载yaml文件
    :param config_path: yaml文件路径
    :return: 返回一个字典，以键值对的方式存储yaml文件中的内容
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def loadTrainConfig(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    加载训练配置，并同时加载训练配置里记录的环境配置和智能体配置
    :param config_path: 训练配置文件路径
    :return: 返回一个三元组，分别是训练配置、环境配置和智能体配置
    """
    config = loadYaml(config_path)
    env_config = loadYaml(config['env_config'])
    agent_config = loadYaml(config['agent_config'])
    return config, env_config, agent_config



class MemoryReplayBuffer:
    """
    经验回放池，RL算法中的一个重要组成部分，如果不使用经验回放池，那么每次训练都只能使用当前的经验，或者本次与环境互动的经验
    如果不用的话，就好像你学习时每天都只能看今天的课本，而不能回顾昨天的知识，这样学习效率会大大降低
    因此RL算法会把很多智能体与环境的互动以(s, a, r, s', is_terminate)的形式存储起来，然后在训练时从存储中随机采样一部分数据进行训练
    注意A2C算法的训练只需要这5个数据以及模型就够了
    """
    def __init__(self, buffer_size: int):
        """
        :param buffer_size: 经验回放池的容量，经验回放池本质是一个list容器，当容器满了之后，就会把最早的经验丢弃掉
        """
        super().__init__()
        self.buffer = []
        self.buffer_size = buffer_size

    def append(self, memory: MemoryTuple):
        """
        向经验回放池中添加一条经验
        :param memory: 一条经验
        :return:
        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(memory)

    def sampleBatch(self, batch_size: int) -> TensorTuple:
        """
        从经验回放池中随机采样一批经验
        :param batch_size: 采样的批次大小
        :return: 详情请看MemoryTuple.makeBatch
        """
        batch_data = random.sample(self.buffer, batch_size)
        return MemoryTuple.makeBatch(batch_data)

    def __len__(self):
        """ 返回经验回放池中的经验数量 """
        return len(self.buffer)


class MovingAverage:
    """
    移动平均，用于计算一段时间内的平均值，比如最近100次的平均值
    RL算法的loss，以及与环境互动时每个episode获得的reward都是波动较大的，因此需要计算一段时间内的平均值，以便于观察训练效果
    """
    def __init__(self, window_size: int):
        """
        :param window_size: 计算平均值的窗口大小，比如你要一直记录最近100次的平均值，那么窗口大小就是100
        """
        self.window_size = window_size
        self.count = 0  # 记录窗口内的数量
        self.sum = 0.0  # 记录窗口内的和

    def append(self, value: float):
        """
        向移动平均中添加一个值
        :param value: 要添加的值
        :return:
        """
        if self.count < self.window_size:
            self.count += 1
        else:
            self.sum -= self.sum / self.window_size
        self.sum += value

    @property
    def average(self):
        """
        计算当前窗口内的平均值
        """
        if self.count == 0:
            return 0.0
        else:
            return self.sum / self.count