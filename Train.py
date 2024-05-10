import cv2  # cv2库用于可视化

# 引入环境，智能体以及训练相关的工具函数
from Environment.GameEnv import GameEnv
from Agents.A2CAgent import A2CAgent
from TrainUtils import *

# 引入tensorboard，它提供了一个web服务器，可以在浏览器中实时监控训练过程
# 它支持记录标量，图像等信息，并可以画出漂亮的实时统计图
from torch.utils.tensorboard import SummaryWriter

# 引入datetime库，用于记录训练开始的时间
# 每次训练我们都会想要存储训练的模型，训练日志等等信息
# 这些信息都会存储在一个以时间命名的文件夹中，准确地讲，Runs/YYYYMMDD_HHMMSS/...
from datetime import datetime
import os

def train():
    # 加载配置文件
    train_config, env_config, agent_config = loadTrainConfig("TrainConfig.yaml")

    # 初始化环境和智能体
    env = GameEnv(env_config)
    agent = A2CAgent(agent_config)

    # 初始化经验回放池
    memory_replay_buffer = MemoryReplayBuffer(train_config['replay_buffer_size'])

    # 初始化统计量，用于记录和可视化
    avg_value_loss = MovingAverage(train_config['moving_average_window_size'])
    avg_policy_loss = MovingAverage(train_config['moving_average_window_size'])
    avg_total_loss = MovingAverage(train_config['moving_average_window_size'])
    avg_episode_reward = MovingAverage(train_config['moving_average_window_size'])
    avg_survive_step = MovingAverage(train_config['moving_average_window_size'])

    # 从配置文件中读取训练相关的参数
    batch_size = train_config['batch_size']
    save_interval = train_config['save_interval']
    log_interval = train_config['log_interval']
    target_model_update_freq = train_config['target_model_update_freq']

    # 新建一个文件夹，用于存储训练过程中的模型，日志等信息，以时间命名
    program_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join("Runs", program_start_time)
    os.makedirs(save_folder, exist_ok=True)

    # 初始化tensorboard
    writer = SummaryWriter(save_folder)

    # train_it是总的训练计数器，我们用它来控制模型保存，日志打印等操作的频率
    train_it = 0

    while True:
        # 第一步，与环境互动获得数据
        agent.eval()    # 设置agent为eval模式，因为此时我们只是想让智能体与环境互动以收集数据，而不是训练
        env.reset()     # 重置环境
        is_terminate = False
        # 与环境互动，直到游戏结束或者达到最大步数，最大步数是为了防止智能体陷入死循环，或者陷入长时间的循环状态，
        # 这样的话数据的收集效率会比较低
        while not is_terminate and env.t < train_config['max_step']:
            # 获取当前state，根据state选择action（别忘了这个action有一定概率是随机的）
            state = env.current_state
            action = agent.getAction(state)
            # 与环境互动，获得下一个state，reward，以及游戏是否结束的标志
            next_state, reward, is_terminate = env.step(action)
            # 将(s, a, r, s', is_terminate)存储到经验回放池中
            # 别忘了我们的MemoryTuple就是存储这5个数据的类
            memory_replay_buffer.append(MemoryTuple(state, action, reward, next_state, is_terminate))

            # 可视化游戏的进程，这里的可视化可以让你监控智能体行为是否正常
            # 比如它是否在死循环，是否在一直走圈圈，是否在一直撞墙，是否总是喜欢往一个方向走等等
            env.render(f"r={env.total_reward:.2f}")
            env.current_state.render()
            cv2.waitKey(1)

        # 记录episode的reward和存活步数
        # 这两个数据是我们最关心的，因为RL任务中，很多时候看loss不够直观，有时候loss根本不下降，或者波动性太大
        # 而这两个数据直接反映了智能体的效果，所以我们会将它们记录下来
        avg_episode_reward.append(env.total_reward)
        avg_survive_step.append(env.t)

        # 中间步，等到经验回放池中有足够的数据
        # 如果回方池中数据不够多，我们就不训练，直接进入下一轮与环境互动继续收集数据
        # 如果数据不足时训练，那么会导致每次的训练数据总是那么几个，我不想让模型拟合这几个数据那么多次
        if len(memory_replay_buffer) < batch_size * 10:
            train_it += 1
            continue

        # 第二步，从经验回放池中采样
        # 此时正是训练的时候，所以我们要将agent设置为train模式
        agent.train()
        # 循环多次，每次循环都求一个batch的数据训练，要注意，最好别只训练一次
        # 因为在上一步，智能体与环境互动一episode，可能收集了几十个甚至几百个数据，假设我们以64的batch_size训练
        # 那么可能有很多的数据都没有用上，的数据利用率就很低
        for _ in range(train_config['updates_per_episode']):
            # 从经验回放池中采样一个batch的数据
            batch_data = memory_replay_buffer.sampleBatch(batch_size)
            # 训练智能体，获得loss
            policy_loss, value_loss, total_loss = agent.update(batch_data)

            # 记录loss
            avg_policy_loss.append(policy_loss)
            avg_value_loss.append(value_loss)
            avg_total_loss.append(total_loss)


        # 第三步，记录，保存
        if train_it % log_interval == 0:
            # 先打印信息，这个信息帮助我们实时监控训练，确保数值稳定没有错误
            # 值得注意的是，就是这里，在第一次训练的时候，我发现value_loss比policy_loss小10倍左右
            # 因此我才在A2CAgent.py的update函数中将value_loss乘以了10
            print(f"it: {train_it}, policy_loss: {avg_policy_loss.average:.5f}, value_loss: {avg_value_loss.average:.5f}, "
                  f"total_loss: {avg_total_loss.average:.5f}, episode_reward: {avg_episode_reward.average:.5f}, "
                  f"survive_step: {avg_survive_step.average:.5f}")
            # tensorboard记录各项指标
            writer.add_scalar('loss/policy_loss', avg_policy_loss.average, train_it)
            writer.add_scalar('loss/value_loss', avg_value_loss.average, train_it)
            writer.add_scalar('loss/total_loss', avg_total_loss.average, train_it)
            writer.add_scalar('reward/episode_reward', avg_episode_reward.average, train_it)
            writer.add_scalar('reward/survive_step', avg_survive_step.average, train_it)

        # 这里存储模型
        if train_it % save_interval == 0:
            agent.save(f"{save_folder}/Model_it_{train_it}")

        # 这里更新智能体的target_model，target_model是用来计算V(s')的
        if train_it % target_model_update_freq == 0:
            agent.updateTargetModel()
            print("Update Target Model")

        train_it += 1


if __name__ == '__main__':
    train()
