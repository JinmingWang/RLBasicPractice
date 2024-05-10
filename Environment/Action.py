import torch.nn
import random
from Environment.EnvUtils import *

class Action:
    action_space = "↑↗→↘↓↙←↖□"
    agent_step_filters = torch.nn.ModuleList([torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1) for _ in range(9)])
    agent_step_filters.to(device)
    agent_step_filters[0].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[0].weight.data[0, 0, 2, 1] = 1
    agent_step_filters[1].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[1].weight.data[0, 0, 2, 0] = 1
    agent_step_filters[2].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[2].weight.data[0, 0, 1, 0] = 1
    agent_step_filters[3].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[3].weight.data[0, 0, 0, 0] = 1
    agent_step_filters[4].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[4].weight.data[0, 0, 0, 1] = 1
    agent_step_filters[5].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[5].weight.data[0, 0, 0, 2] = 1
    agent_step_filters[6].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[6].weight.data[0, 0, 1, 2] = 1
    agent_step_filters[7].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[7].weight.data[0, 0, 2, 2] = 1
    agent_step_filters[8].weight.data = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
    agent_step_filters[8].weight.data[0, 0, 1, 1] = 1
    agent_step_filters.requires_grad_(False)

    directions = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [0, 0]])

    def __init__(self, action: Union[int, torch.Tensor]) -> None:
        """
        Action类，用于表示一个动作
        :param action: 可以是动作对应的数字，也可以以torch.Tensor的格式给数字，Union类型注释表示可以是多中类型中的一个
        """
        # 如果是数字，转换成torch.Tensor
        if isinstance(action, int):
            action = torch.tensor([action])
        # 把tensor转换成long类型，放到CUDA上，后面的view(1)是为了把这个数字张量的尺寸变成(1,)
        # 这样的话，在后面makeBatch的时候，可以直接使用torch.cat把很多action拼接成尺寸为(B,)的张量
        self.action = action.to(device).to(torch.long).view(1)

    def __str__(self) -> str:
        return self.action_space[self.action.item()]
    
    def __repr__(self) -> str:
        return self.action_space[self.action.item()]


    def applyTo(self, game_env: "GameEnv"):
        game_env.grid_world[..., 0] = self.agent_step_filters[self.action.item()](
            game_env.grid_world[..., 0].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        game_env.agent_pos += self.directions[self.action]

    @staticmethod
    def randomAction() -> "Action":
        """ 随机生成一个动作，注意random.randint是包含两端的 """
        return Action(random.randint(0, 8))
    
    @staticmethod
    def makeBatch(actions: List["Action"]) -> torch.Tensor:
        """
        把一系列的Action拼接成一个batch
        :param actions: 一系列的Action
        :return: 形如(B,)的张量
        """
        return torch.cat([action.action for action in actions], dim=0)