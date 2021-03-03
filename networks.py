import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_xavier


# Suggestion od Mareka
class networkV1(nn.Module):
    def __init__(self, actions_count):
        super(networkV1, self).__init__()
        self.actions_count = actions_count

        self.conv1s = nn.Conv2d(4, 16, 3, stride=1, padding=1)  # zmen 16 => 32
        self.conv2s = nn.Conv2d(16, 32, 3, stride=1,
                                padding=1)  # zmen 16 => 32
        self.conv3s = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4s = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # 1 konvolucna a 2 fully connected - rovnako velke
        # 512 pozri na marekovom githube
        self.fc1s = nn.Linear(5 * 5 * 64, 128)
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

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1s(x))
        x = F.relu(self.fc2s(x))

        outActor = F.relu(self.fcp(x))
        outCritic = F.relu(self.fcv(x))
        return outActor, outCritic


# DeepMind - atari DQN https://arxiv.org/pdf/1312.5602v1.pdf
class networkV2(nn.Module):
    def __init__(self, actions_count):
        super(networkV2, self).__init__()
        self.actions_count = actions_count

        self.conv1s = nn.Conv2d(4, 16, 8, stride=4, padding=0)
        self.conv2s = nn.Conv2d(16, 32, 4, stride=3, padding=0)

        self.fc1s = nn.Linear(6 * 6 * 32, 256)

        self.fcp = nn.Linear(256, actions_count)
        self.fcv = nn.Linear(256, 1)

        self.apply(weights_init_xavier)  # TODO zisti ako presne funguje

    def forward(self, x):
        x = F.relu(self.conv1s(x))
        x = F.relu(self.conv2s(x))

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1s(x))

        outActor = F.relu(self.fcp(x))
        outCritic = F.relu(self.fcv(x))
        return outActor, outCritic


# Upravene po konzultacii
class networkV3(nn.Module):
    def __init__(self, actions_count):
        super(networkV3, self).__init__()
        self.actions_count = actions_count

        self.conv1s = nn.Conv2d(
            4, 32, 3, stride=1, padding=1)  # zmena 16 => 32
        self.conv2s = nn.Conv2d(32, 32, 3, stride=1,
                                padding=1)  # zmena 16 => 32
        self.conv3s = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4s = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fca1 = nn.Linear(5 * 5 * 64, 512)  # 512
        self.fcc1 = nn.Linear(5 * 5 * 64, 512)  # 512

        self.fca2 = nn.Linear(512, actions_count)
        self.fcc2 = nn.Linear(512, 1)

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

        x = x.flatten(start_dim=1)

        x_a = F.relu(self.fca1(x))
        x_c = F.relu(self.fcc1(x))

        outActor = F.relu(self.fca2(x_a))
        outCritic = F.relu(self.fcc2(x_c))
        return outActor, outCritic
