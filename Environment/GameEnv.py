"""
这个文件定义了环境类，也就是游戏

"""

from Environment.EnvUtils import *
from Environment.State import State
from Environment.Action import Action


class GameEnv:
    AGENT = 0
    GOAL = 1
    # 格子世界有4个边缘，那么本质上有4种障碍物，分别是向上下左右运动的障碍物，这里的NESW是指障碍物运动的方向
    # 他们的出生方向是相反的，比如障碍物向上运动，那么它就是从下边界出生的
    # 不要小看这里的注释，不然到项目后期你很可能会搞不清楚这个方向，例如：（诶？我这个障碍物在下面出生为啥是N？）
    OBSTACLE_N = 2
    OBSTACLE_E = 3
    OBSTACLE_S = 4
    OBSTACLE_W = 5

    # 这里定义了卷积操作，该卷积可以将障碍物按照他们的运动方向平移一格
    obstacle_step_filter = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False, groups=4)
    obstacle_step_filter.weight.data = torch.zeros(4, 1, 3, 3, device=device, dtype=torch.float32)
    obstacle_step_filter.weight.data[0, 0, 2, 1] = 1  # 上
    obstacle_step_filter.weight.data[1, 0, 1, 0] = 1  # 右
    obstacle_step_filter.weight.data[2, 0, 0, 1] = 1  # 下
    obstacle_step_filter.weight.data[3, 0, 1, 2] = 1  # 左
    obstacle_step_filter.requires_grad_(False)


    def __init__(self, env_config: Dict[str, Any]) -> None:
        """初始化游戏类

        1. 格子世界：游戏的世界是一个二维的格子，每个格子中可以有智能体、目标、障碍物等
        2. 智能体：初始化在世界中任意非边缘的地方，智能体可以向周围8个方向移动或者不动
        3. 障碍物：初始化在世界中边缘，朝着它对面的方向移动，每隔一定时间就会生成一波障碍物

        :param env_config: 游戏的配置，包括游戏的大小、障碍物的密度等
        """
        # 6个通道分别是：智能体，目标，障碍物（向上下左右运动）
        self.height: int = env_config["grid_world_height"]
        self.width: int = env_config["grid_world_width"]
        self.channels: int = 6

        # 障碍物的密度和生成频率
        self.obstacle_prob: float = env_config["obstacle_prob"]
        self.obstacle_freq: float = env_config["obstacle_freq"]
        self.n_foods: int = env_config["food_num"]

        self.tensor_size: Tuple[int, int, int] = (self.height, self.width, self.channels)
        self.grid_world: torch.Tensor = torch.zeros(*self.tensor_size, dtype=torch.float32, device=device)
        self.agent_pos: RowCol = None
        self.target_pos_list: List[RowCol] = []
        self.current_state: State = None

        self.t: int = 0  # 记录游戏的时间，用来控制障碍物的生成频率
        self.total_reward: float = 0  # 记录游戏的总奖励

        self.reset()

    def reset(self):
        """
        ！！！reset函数非常重要，在episodic任务中，每次开始新的episode时，都要调用这个函数。
        episodic任务是指那种有明确开始和结束的任务，比如玩一局游戏，或者走到终点等。
        """
        self.grid_world = torch.zeros(*self.tensor_size, dtype=torch.float32, device=device)
        # 随机初始化智能体位置
        self.agent_pos = np.random.randint(1, (self.height-1, self.width-1))
        self.grid_world[self.agent_pos[0], self.agent_pos[1], self.AGENT] = 1
        # 随机初始化目标位置
        self.target_pos_list = []
        self.generateTarget()
        # 随机初始化障碍物
        self.generateObstacles()
        self.current_state = State(self)

        self.t = 0
        self.total_reward = 0


    def generateObstacles(self):
        """这个函数专门为了生成障碍物"""
        # 生成上边界的障碍物
        row = 0
        obstacles = torch.rand(self.width) < self.obstacle_prob
        self.grid_world[row, :, self.OBSTACLE_S] = obstacles

        # 生成下边界的障碍物
        row = self.height-1
        obstacles = torch.rand(self.width) < self.obstacle_prob
        self.grid_world[row, :, self.OBSTACLE_N] = obstacles

        # 生成左边界的障碍物
        col = 0
        obstacles = torch.rand(self.height) < self.obstacle_prob
        self.grid_world[:, col, self.OBSTACLE_E] = obstacles

        # 生成右边界的障碍物
        col = self.width-1
        obstacles = torch.rand(self.height) < self.obstacle_prob
        self.grid_world[:, col, self.OBSTACLE_W] = obstacles


    def generateTarget(self):
        while len(self.target_pos_list) < self.n_foods:
            # 确保所有目标位置和智能体位置不一样，一样的话就重新随机一个
            target_pos = np.random.randint(1, (self.height - 1, self.width - 1))
            while np.all(target_pos == self.agent_pos):
                target_pos = np.random.randint(1, (self.height - 1, self.width - 1))
            self.target_pos_list.append(target_pos)
            self.grid_world[target_pos[0], target_pos[1], self.GOAL] = 1


    def step(self, action: Action) -> Tuple[State, float, bool]:
        """
        这个函数是游戏的核心，也是RL算法和游戏之间的接口
        :param action: 要执行的动作
        :return: 返回新的state，reward，和一个布尔值表示游戏是否结束
        """

        reward = 0
        is_terminate = False

        # 这里更新智能体的位置
        action.applyTo(self)

        # 这里更新障碍物的位置
        self.grid_world[..., self.OBSTACLE_N:] = \
            self.obstacle_step_filter(self.grid_world[..., self.OBSTACLE_N:].permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)

        # 这里检测是否已经跑出边界，如果是，返回state，reward，isTerminate=True
        if self.isOutOfBoundary(self.agent_pos):
            reward -= 2
            self.t += 1
            # 这里更新state
            self.current_state = State(self)
            self.total_reward += reward
            return self.current_state, reward, True

        # 这里检测是否碰到目标
        for ti, target_pos in enumerate(self.target_pos_list):
            if np.all(self.agent_pos == target_pos):
                reward += 1
                self.grid_world[self.agent_pos[0], self.agent_pos[1], self.GOAL] = 0
                self.target_pos_list.pop(ti)
                self.generateTarget()
                break

        # 这里检测是否碰到障碍物
        if self.isObstacle(self.agent_pos):
            reward -= 1
            is_terminate = True

        self.t += 1
        self.total_reward += reward

        # 这里检测是否该生成新的障碍物
        if self.t % self.obstacle_freq == 0:
            self.generateObstacles()

        # 这里更新state
        self.current_state = State(self)

        return self.current_state, reward, is_terminate


    def render(self, msg=None):
        """
        这个函数专门为了显示游戏画面
        就显示一张彩色图片吧，智能体是蓝色，目标是绿色，障碍物是红色
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[..., 0] = self.grid_world[..., self.AGENT].cpu().numpy() * 255
        canvas[..., 1] = self.grid_world[..., self.GOAL].cpu().numpy() * 255
        canvas[..., 2] = np.max(self.grid_world[..., self.OBSTACLE_N:].cpu().numpy(), axis=-1) * 255
        ratio = 6
        canvas = cv2.resize(canvas, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        # 画以智能体为中心的正方形
        pos_xy = self.agent_pos[::-1]
        cv2.rectangle(canvas, (pos_xy-State.HALF_SIZE)*ratio, (pos_xy+State.HALF_SIZE+1)*ratio, (255, 255, 255), 1)
        if msg is not None:
            cv2.putText(canvas, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Game", canvas)


    def isOutOfBoundary(self, pos: RowCol) -> bool:
        """
        检测是否出界
        :param pos: 要检测的位置
        :return: 布尔值，是否出界
        """
        return pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width

    def isObstacle(self, pos: RowCol) -> bool:
        """
        检测某位置是否存在障碍物
        :param pos: 要检测的位置
        :return: 布尔值，是否存在障碍物
        """
        return torch.any(self.grid_world[pos[0], pos[1], self.OBSTACLE_N:] > 0).item()


def humanControlTest():
    # 单元测试是相当重要的，量变产生质变，堆的屎山太多了会逐渐变成无法测试的一坨，所以要趁还没发展到那个地步的时候就开始写单元测试
    # 这个测试先看一下参数是否正确，环境初始化和reset是否对劲，
    env = GameEnv({
        "grid_world_height": 128,
        "grid_world_width": 128,
        "obstacle_prob": 0.1,
        "obstacle_freq": 20,
        "food_num": 10
    })

    is_terminate = False

    k = 0
    control_dict = {
        ord("w"): 0, ord("e"): 1, ord("d"): 2, ord("c"): 3,
        ord("x"): 4, ord("z"): 5, ord("a"): 6, ord("q"): 7,
        ord("s"): 8
    }

    env.render()
    env.current_state.render()

    while not is_terminate:
        if k in control_dict:
            action = control_dict[k]
        elif k == ord(" "):
            break
        else:
            k = cv2.waitKey(0)
            continue
        next_step, reward, is_terminate = env.step(Action(action))
        print("reward=", reward)
        env.render()
        env.current_state.render()
        k = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    humanControlTest()