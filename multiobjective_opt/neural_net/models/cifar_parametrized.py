import torch.nn as nn
import torch.nn.functional as F


# Базовый блок ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Добавляем shortcut connection
        out = F.relu(out)
        return out


# Модель ResNet
class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,  # Количество блоков в каждом слое
        num_filters=64,  # Количество фильтров в начальном слое
        use_batchnorm=True,  # Использовать BatchNorm
        use_dropout=False,  # Использовать Dropout
        dropout_prob=0.5,  # Вероятность Dropout
        num_classes=10,  # Количество классов (для CIFAR-10)
    ):
        super(ResNet, self).__init__()
        self.in_channels = num_filters
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # Начальный слой
        self.conv1 = nn.Conv2d(
            3, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters) if use_batchnorm else nn.Identity()
        self.layer1 = self._make_layer(
            num_filters, num_blocks[0], stride=1, use_batchnorm=use_batchnorm
        )
        self.layer2 = self._make_layer(
            num_filters * 2, num_blocks[1], stride=2, use_batchnorm=use_batchnorm
        )
        self.layer3 = self._make_layer(
            num_filters * 4, num_blocks[2], stride=2, use_batchnorm=use_batchnorm
        )
        self.layer4 = self._make_layer(
            num_filters * 8, num_blocks[3], stride=2, use_batchnorm=use_batchnorm
        )
        self.linear = nn.Linear(num_filters * 8, num_classes)
        self.dropout = nn.Dropout(dropout_prob) if use_dropout else nn.Identity()

    def _make_layer(self, out_channels, num_blocks, stride, use_batchnorm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                BasicBlock(self.in_channels, out_channels, stride, use_batchnorm)
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # Global Average Pooling
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
