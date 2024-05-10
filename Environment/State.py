from Environment.EnvUtils import *

class State:
    WINDOW_SIZE = 33
    HALF_SIZE = 16
    def __init__(self, game_env: "GameEnv") -> None:
        """
        初始化状态类
        :param 游戏环境类
        """

        # 从game_env.grid_world中截取以game_env.agent_pos为中心，前后各16个像素的图像
        # 这里的截取是为了让智能体能够看到周围的环境
        # 要考虑到截取的时候，可能会超出边界的情况，因此先在边界外填充厚度为16，值为-1的一圈
        background = - torch.ones(game_env.height + self.HALF_SIZE * 2 + 2, game_env.width + self.HALF_SIZE * 2 + 2,
                                  game_env.channels, dtype=torch.float32, device=device)
        background[self.HALF_SIZE+1:-self.HALF_SIZE-1, self.HALF_SIZE+1:-self.HALF_SIZE-1, :] = game_env.grid_world

        area_center = game_env.agent_pos
        start_row = area_center[0] + 1
        start_col = area_center[1] + 1
        end_row = start_row + State.WINDOW_SIZE
        end_col = start_col + State.WINDOW_SIZE
        self.tensor = background[start_row:end_row, start_col:end_col, :].permute(2, 0, 1)  # (6, 33, 33)
        if self.tensor.shape[1] != self.WINDOW_SIZE or self.tensor.shape[2] != self.WINDOW_SIZE:
            raise Exception("State tensor shape error")

    def render(self):
        tensor = self.tensor / 2 + 0.5
        canvas = np.zeros((self.WINDOW_SIZE, self.WINDOW_SIZE, 3), dtype=np.uint8)
        canvas[..., 0] = tensor[0, ...].cpu().numpy() * 255
        canvas[..., 1] = tensor[1, ...].cpu().numpy() * 255
        canvas[..., 2] = np.max(tensor[2:, ...].cpu().numpy(), axis=0) * 255
        ratio = 8
        canvas = cv2.resize(canvas, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("State", canvas)

    @staticmethod
    def makeBatch(states: List["State"]) -> torch.Tensor:
        """
        将状态列表转换为batch
        :param states: 状态列表
        :return: 状态的batch，尺寸为(B, 6, 33, 33)
        """
        return torch.stack([state.tensor for state in states])





