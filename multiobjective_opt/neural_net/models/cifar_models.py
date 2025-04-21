import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(32 * 32 * 3, 10)  # 32x32x3 -> 10 классов

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Преобразуем в 1D
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)



class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CNNDropout(nn.Module):
    def __init__(self):
        super(CNNDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNNBatchNorm(nn.Module):
    def __init__(self):
        super(CNNBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Остаточный блок
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_shortcut = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.with_shortcut = with_shortcut
        self.shortcut = nn.Sequential()
        if with_shortcut:
            if (stride != 1 or in_channels != out_channels):
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.with_shortcut:
            out += self.shortcut(x)  # Добавляем shortcut connection
        out = F.relu(out)
        return out


# Модель ResNet
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, with_shortcut = True):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Начальный слой
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, with_shortcut= with_shortcut)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, with_shortcut= with_shortcut)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, with_shortcut= with_shortcut)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, with_shortcut= with_shortcut)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, with_shortcut):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride, with_shortcut=with_shortcut))
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
        out = self.linear(out)
        return out


# Создание модели ResNet-18
def ResNet18(with_shortcut = True):
    return ResNet([2, 2, 2, 2], with_shortcut=with_shortcut)  # ResNet-18 имеет 4 слоя с 2 блоками в каждом


# cool models

import torch
import torchvision.models as models
import torch.nn as nn

# Download and initialize ResNet18
def ResNet18Torch():
    resnet18 = models.resnet18(pretrained=False)
    resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18.maxpool = nn.Identity()  # Remove maxpool
    resnet18.fc = nn.Linear(512, 10)
    return resnet18

# # Calculate total number of parameters
# total_params = sum(p.numel() for p in resnet18.parameters())

# print(f"Total number of parameters in ResNet18: {total_params:,}")
# sample_input = torch.randn(1, 3, 32, 32)  # One sample CIFAR10 image
# output = resnet18(sample_input)
# print(f"Output shape: {output.shape}")


class MLPModule(nn.Module):
    def __init__(self, in_params, out_params, with_norm= True):
        super(MLPModule, self).__init__()
        layers = [nn.Linear(in_params, out_params)]
        if with_norm:
            layers.append(nn.BatchNorm1d(out_params))
        layers.append(nn.ReLU())
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DeepMLP(nn.Module):
    def __init__(self, with_norm = True):
        super(DeepMLP, self).__init__()

        layers = [nn.Flatten(),
                MLPModule(3072, 1024, with_norm)
        ]
        for _ in range(8):
            layers.append(MLPModule(1024, 1024, with_norm))

        layers.append(nn.Linear(1024, 10))
        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.layers(x)


class ShallowMLP(nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3072, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 10)  # CIFAR10 has 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class VGGLike(nn.Module):
    def __init__(self):
        super(VGGLike, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, patch_size=4, embed_dim=336, num_heads=8, num_layers=8):
        super(ViTModel, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2  # CIFAR10 is 32x32

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        # MLP head
        self.mlp_head = nn.Linear(embed_dim, 10)  # 10 classes for CIFAR10

        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # Patch embedding: (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        x = self.blocks(x)

        # Take cls token output
        x = x[:, 0]

        # Classification head
        x = self.mlp_head(x)
        return x
