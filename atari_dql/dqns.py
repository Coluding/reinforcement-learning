import torch
import torch.nn as nn
from typing import Sequence


class DQNCNN(nn.Module):
    def __init__(self,
                 conv_layer_sizes=((32, 6, 4), (64, 4, 2), (64, 3, 1)),
                 input_shape: Sequence[int] = (4,84,84),
                 n_actions: int = 4,
                 init_weights: bool = True) -> None:
        super(DQNCNN, self).__init__()

        if conv_layer_sizes is None:
            conv_layer_sizes = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        self.input_shape = input_shape

        conv_layer_list = []
        in_ch = input_shape[0]
        for out_ch, kernel_size, stride in conv_layer_sizes:
            conv_layer_list.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride))
            conv_layer_list.append(nn.ReLU())
            in_ch = out_ch

        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(-3, -1)
        self.classifier = nn.Sequential(
            nn.Linear(self._compute_final_conv_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        if init_weights:
            self._init_weights()

    def _compute_final_conv_dim(self) -> int:
        x = torch.unsqueeze(torch.randn(self.input_shape), 0)
        x = self.conv_layer(x)
        return torch.prod(torch.tensor(x.size())).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.flatten(x)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def copy_weights(self, other: nn.Module) -> None:
        self.load_state_dict(other.state_dict())