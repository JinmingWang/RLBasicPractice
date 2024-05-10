import numpy as np
import matplotlib.pyplot as plt
# plot 3D
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    world = np.zeros((6, 10, 16), dtype=np.float32)
    world[0, 2, 5] = 1.0
    world[1, 4, 8] = 1.0
    world[2, -1, [2, 4, 9, 10, 12]] = 1.0
    world[3, [1, 6], 0] = 1.0
    world[4, 0, -2] = 1.0
    world[5, [3, 4, 5, 8], -1] = 1.0

    titles = ["Agent", "Goal", "Obstacle ↑", "Obstacle →", "Obstacle ↓", "Obstacle ←"]
    cmaps = ["Blues", "Greens", "Reds", "Reds", "Reds", "Reds"]

    for row in range(2):
        for col in range(3):
            plt.subplot(2, 3, row * 3 + col + 1)
            plt.title(titles[row * 3 + col])
            plt.imshow(world[row * 3 + col], cmap=cmaps[row * 3 + col])
    plt.show()

    # plot 3D, every channel in world is a layer
    Y = np.arange(10).repeat(16)
    X = np.arange(16).repeat(10).reshape(16, 10).T.reshape(-1)
    ax = plt.axes(projection='3d')

    # plot agent_channel to 3D world
    Z = np.zeros_like(X)
    agent_channel = world[0].reshape(-1) / 2 + 0.5    # (160)
    ax.scatter3D(X, Y, Z, c=agent_channel, marker="s", s=50, alpha=agent_channel, cmap='Blues')

    # plot goal_channel to 3D world
    Z = np.ones_like(X)
    goal_channel = world[1].reshape(-1) / 2 + 0.5     # (160)
    ax.scatter3D(X, Y, Z, c=goal_channel, marker="s", s=50, alpha=goal_channel, cmap='Greens')

    # plot other channels to 3D world
    for i in range(2, 6):
        Z = np.ones_like(X) * i
        channel = world[i].reshape(-1) / 2 + 0.5      # (160)
        ax.scatter3D(X, Y, Z, c=channel, marker="s", s=50, alpha=channel, cmap='Reds')

    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    ax.set_zlabel('Channel')
    plt.show()



