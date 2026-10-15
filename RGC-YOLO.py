class SafeDepthwiseConv2d(nn.Conv2d):

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        if (self.groups <= 1 or x.device.type != "cuda"
                or not torch.backends.cudnn.is_available()):
            return super().forward(x)
        with torch.backends.cudnn.flags(enabled=False):
            return functional.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1):
        padding = kernel_size // 2
        conv = (SafeDepthwiseConv2d if groups > 1 else nn.Conv2d)(
            in_channels, out_channels, kernel_size, 1, padding, groups=groups, bias=False,
        )
        super().__init__(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class GLC(nn.Module):

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 4):
        super().__init__()
        channels = c1 if c2 is None else c2
        self.input_projection = ConvNormAct(c1, channels, 1) if c1 != channels else nn.Identity()
        hidden_channels = max(8, channels // reduction)
        self.local_path = nn.Sequential(
            ConvNormAct(channels, channels, 5, groups=channels),
            ConvNormAct(channels, channels, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.input_projection(inputs)
        local_features = self.local_path(inputs)
        gate = self.channel_gate(inputs) * self.spatial_gate(inputs)
        return inputs + self.gamma * local_features * gate



class RCA(nn.Module):

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        ch = c1 if c2 is None else c2
        self.conv1 = nn.Conv2d(c1, ch, 1, bias=False)
        self.dw = SafeDepthwiseConv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.conv2 = nn.Conv2d(ch, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        hidden = max(ch // 8, 8)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Linear(ch, hidden)
        self.se_act = nn.SiLU()
        self.se_fc2 = nn.Linear(hidden, ch)
        self.se_sig = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.dw(y)
        y = self.bn(y)
        b, c, _, _ = y.shape
        w = self.se_pool(y).view(b, c)
        w = self.se_fc(w)
        w = self.se_act(w)
        w = self.se_fc2(w)
        w = self.se_sig(w).view(b, c, 1, 1)
        y = y * w
        y = self.conv2(y)
        return x + self.gamma * y
