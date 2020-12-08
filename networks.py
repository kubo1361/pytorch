import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_xavier


class networkV1(nn.Module):
    def __init__(self, actions_count):
        super(networkV1, self).__init__()

        self.conv1s = nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.conv2s = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3s = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4s = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1s = nn.Linear(6 * 6 * 64, 256)
        self.fc2s = nn.Linear(128, 256)

        self.fcp = nn.Linear(256, actions_count)
        self.fcv = nn.Linear(256, 1)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1s(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv2s(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv3s(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = F.relu(self.conv4s(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = x.view(-1, 2304)

        x = F.relu(self.fc1s(x))
        x = F.relu(self.fc2s(x))

        policy = F.relu(self.fcp(x))
        value = F.relu(self.fcv(x))
        return policy, value
