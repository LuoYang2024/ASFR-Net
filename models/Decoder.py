import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class GroupedContextExtractor(nn.Module):
    def __init__(self, channels, groups=4):
        super(GroupedContextExtractor, self).__init__()
        self.groups = groups
        self.split_d = channels // groups
        self.conv_groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.split_d, self.split_d, 1, bias=False), nn.BatchNorm2d(self.split_d),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.split_d, self.split_d, kernel_size=3, padding=i + 1, dilation=i + 1, groups=self.split_d,
                          bias=False),
                nn.BatchNorm2d(self.split_d), nn.ReLU(inplace=True)
            ) for i in range(groups)
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_split = x.view(B, self.groups, self.split_d, H, W)
        out_groups = [self.conv_groups[i](x_split[:, i, :, :, :]) for i in range(self.groups)]
        return self.fusion_conv(torch.cat(out_groups, dim=1))


class SimplifiedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(SimplifiedFusionBlock, self).__init__()
        self.high_context_extractor = GroupedContextExtractor(channels, groups=4)
        self.low_feature_enhancer = CoordAtt(channels, channels)
        self.gate_generator = nn.Sequential(nn.Conv2d(channels * 2, 2, 3, padding=1), nn.Softmax(dim=1))
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.cls = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x_low, x_high):
        target_size = x_low.shape[-2:]
        x_high_up = F.interpolate(x_high, size=target_size, mode='bilinear', align_corners=False)
        high_context_fused = self.high_context_extractor(x_high_up)
        low_enhanced = self.low_feature_enhancer(x_low)

        gates = self.gate_generator(torch.cat([low_enhanced, high_context_fused], dim=1))
        gate_low, gate_high = gates[:, 0:1], gates[:, 1:2]
        fused_dynamic = low_enhanced * gate_low + high_context_fused * gate_high

        output_feat = self.final_conv(fused_dynamic + x_low)
        return output_feat, self.cls(output_feat)


class SimplifiedDecoder(nn.Module):
    def __init__(self, channels):
        super(SimplifiedDecoder, self).__init__()
        self.decoder_p4 = SimplifiedFusionBlock(channels)
        self.decoder_p3 = SimplifiedFusionBlock(channels)
        self.decoder_p2 = SimplifiedFusionBlock(channels)
        self.decoder_p1 = SimplifiedFusionBlock(channels)
        self.p5_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, c1, c2, c3, c4, c5):
        p5 = self.p5_conv(c5)
        p4, mask_p4 = self.decoder_p4(c4, p5)
        p3, mask_p3 = self.decoder_p3(c3, p4)
        p2, mask_p2 = self.decoder_p2(c2, p3)
        p1, mask_p1 = self.decoder_p1(c1, p2)
        return mask_p1, mask_p2, mask_p3, mask_p4