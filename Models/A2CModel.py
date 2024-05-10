import torch

from Models.ModelUtils import *

class A2CModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (B, 6, 33, 33)
        self.body = nn.Sequential(
            ConvBnReLU(6, 16, 3, 1, 1),  # (B, 16, 33, 33)
            ConvBnReLU(16, 32, 3, 1, 0),  # (B, 32, 31, 31)
            ConvBnReLU(32, 64, 3, 2, 1),  # (B, 64, 16, 16)
            ConvBnReLU(64, 128, 3, 2, 1),  # (B, 128, 8, 8)
            ConvBnReLU(128, 256, 3, 2, 1),  # (B, 256, 4, 4)
            ConvBnReLU(256, 512, 3, 1, 0),  # (B, 512, 2, 2)
            nn.Flatten(),  # (B, 2048)
        )

        self.value_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Flatten(0),  # (B, )
        )

        self.policy_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 9),
            nn.Softmax(dim=1),  # (B, 9)
        )

    def getPolicyParams(self):
        return self.policy_head.parameters()

    def getValueParams(self):
        return [{'params': self.body.parameters()}, {'params': self.value_head.parameters()}]

    def forward(self, x):
        x = self.body(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return policy, value


if __name__ == '__main__':
    model = A2CModel()
    inp = torch.randn(4, 6, 33, 33)
    policy, value = model(inp)
    print(policy.shape, value.shape)
