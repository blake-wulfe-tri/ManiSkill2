import torch

from mani_skill2.dynamics.base import DynamicsModel


class UnetFiLMDynamicsModel(DynamicsModel):
    def __init__(self, n_channels, cond_size, lr=5e-4):
        super().__init__()

        self.n_channels = n_channels
        self.cond_size = cond_size
        self.bilinear = True

        def film(x, gamma, beta):
            gamma = gamma[..., None, None]
            beta = beta[..., None, None]
            return x * gamma + beta

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        class DoubleConvFiLM(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1
                )
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1
                )
                self.bn2 = nn.BatchNorm2d(out_channels)

            def forward(self, x, gamma, beta):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = film(x, gamma, beta)
                x = self.relu(x)
                return x

        class DownFiLM(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.max_pool = nn.MaxPool2d(2)
                self.conv = DoubleConvFiLM(in_channels, out_channels)

            def forward(self, x, gamma, beta):
                x = self.max_pool(x)
                x = self.conv(x, gamma, beta)
                return x

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranpose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1,
                    [
                        diffX // 2,
                        diffX - diffX // 2,
                        diffY // 2,
                        diffY - diffY // 2,
                    ],
                )
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = DownFiLM(64, 128)
        self.down2 = DownFiLM(128, 256)
        self.down3 = DownFiLM(256, 512)
        self.down4 = DownFiLM(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        n_film_parameters = (128 + 256 + 512 + 512) * 2
        self.film_generator = nn.Linear(self.cond_size, n_film_parameters)
        torch.nn.init.kaiming_uniform_(self.film_generator.weight)
        self.film_generator.bias.data.zero_()

    def forward(self, obs, action):
        film_parameters = self.film_generator(action)
        (d1_g, d1_b, d2_g, d2_b, d3_g, d3_b, d4_g, d4_b) = film_parameters.split(
            (128, 128, 256, 256, 512, 512, 512, 512),
            dim=1,
        )

        x1 = self.inc(obs)
        x2 = self.down1(x1, d1_g, d1_b)
        x3 = self.down2(x2, d2_g, d2_b)
        x4 = self.down3(x3, d3_g, d3_b)
        x5 = self.down4(x4, d4_g, d4_b)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
