from torch import nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, h=200, args=None):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(256, 2, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.l1 = nn.Linear(400, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.02)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = x.view(x.shape[0],-1)
        x = F.leaky_relu(self.l1(x), 0.1)
        x = F.leaky_relu(self.l2(x), 0.1)
        x = self.l3(x)
        return x