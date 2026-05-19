import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule_WithFreqResidual(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule_WithFreqResidual, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.freq_enhancer = nn.Sequential(
            nn.Conv2d(out_d, out_d, kernel_size=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, c_fuse, c, freq_feature):
        c_fuse = self.conv_fuse(c_fuse)
        enhanced_freq_feature = self.freq_enhancer(freq_feature)
        return self.relu(c_fuse + self.conv_identity(c) + enhanced_freq_feature)


class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super(FrequencyChannelAttention, self).__init__()
        in_channels = channels * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, complex_features):
        return self.fc(self.avg_pool(complex_features))


class FrequencySpatialAttention(nn.Module):
    def __init__(self, channels):
        super(FrequencySpatialAttention, self).__init__()
        in_channels = channels * 2
        self.feature_processor = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, complex_features):
        processed_features = self.feature_processor(complex_features)
        max_pool = torch.max(processed_features, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(processed_features, dim=1, keepdim=True)
        return self.spatial_conv(torch.cat([max_pool, avg_pool], dim=1))


class AFFM_V4_ParallelAttention(nn.Module):
    def __init__(self, channels=64):
        super(AFFM_V4_ParallelAttention, self).__init__()
        self.channel_att = FrequencyChannelAttention(channels)
        self.spatial_att = FrequencySpatialAttention(channels)

    def forward(self, f1, f2):
        f1_fft = torch.fft.fft2(f1, norm='ortho')
        f2_fft = torch.fft.fft2(f2, norm='ortho')
        diff_fft = f1_fft - f2_fft
        diff_features = torch.cat([diff_fft.real, diff_fft.imag], dim=1)

        channel_gate = self.channel_att(diff_features)
        spatial_gate = self.spatial_att(diff_features)
        total_gate = channel_gate * spatial_gate

        f1_filtered = torch.fft.ifft2(f1_fft * total_gate, s=f1.shape[-2:], norm='ortho').real
        f2_filtered = torch.fft.ifft2(f2_fft * total_gate, s=f2.shape[-2:], norm='ortho').real
        return f1_filtered, f2_filtered


class MultiLevelAFFM_Enhanced(nn.Module):
    def __init__(self, channels, in_dims):
        super(MultiLevelAFFM_Enhanced, self).__init__()
        self.projection_layers = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_dims])
        self.affm_modules = nn.ModuleList([AFFM_V4_ParallelAttention(channels=channels) for _ in range(len(in_dims))])

    def forward(self, f1_list, f2_list):
        f1_projected = [self.projection_layers[i](f1_list[i]) for i in range(len(f1_list))]
        f2_projected = [self.projection_layers[i](f2_list[i]) for i in range(len(f2_list))]
        f1_filtered_list, f2_filtered_list = [], []
        for i in range(len(f1_list)):
            f1_filtered, f2_filtered = self.affm_modules[i](f1_projected[i], f2_projected[i])
            f1_filtered_list.append(f1_filtered)
            f2_filtered_list.append(f2_filtered)
        return tuple(f1_filtered_list), tuple(f2_filtered_list)


class SpatioFrequencyReinforcementModule(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(SpatioFrequencyReinforcementModule, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        self.frequency_processor = MultiLevelAFFM_Enhanced(channels=out_d, in_dims=in_d)

        def blk(c, d, ks, st, pd, mp=1):
            lyr = []
            if mp > 1: lyr.append(nn.MaxPool2d(mp, mp))
            lyr.extend([nn.Conv2d(c, d, ks, st, pd), nn.BatchNorm2d(d), nn.ReLU(inplace=True)])
            if ks == 1: lyr.extend([nn.Conv2d(d, d, 3, 1, 1, groups=d), nn.BatchNorm2d(d), nn.ReLU(inplace=True)])
            return nn.Sequential(*lyr)

        self.conv_scale1_c1 = blk(in_d[0], self.mid_d, 3, 1, 1)
        self.conv_scale2_c1 = blk(in_d[0], self.mid_d, 1, 1, 0, 2)
        self.conv_scale3_c1 = blk(in_d[0], self.mid_d, 1, 1, 0, 4)
        self.conv_scale4_c1 = blk(in_d[0], self.mid_d, 1, 1, 0, 8)
        self.conv_scale5_c1 = blk(in_d[0], self.mid_d, 1, 1, 0, 16)
        self.conv_scale1_c2 = blk(in_d[1], self.mid_d, 1, 1, 0)
        self.conv_scale2_c2 = blk(in_d[1], self.mid_d, 3, 1, 1)
        self.conv_scale3_c2 = blk(in_d[1], self.mid_d, 1, 1, 0, 2)
        self.conv_scale4_c2 = blk(in_d[1], self.mid_d, 1, 1, 0, 4)
        self.conv_scale5_c2 = blk(in_d[1], self.mid_d, 1, 1, 0, 8)
        self.conv_scale1_c3 = blk(in_d[2], self.mid_d, 1, 1, 0)
        self.conv_scale2_c3 = blk(in_d[2], self.mid_d, 1, 1, 0)
        self.conv_scale3_c3 = blk(in_d[2], self.mid_d, 3, 1, 1)
        self.conv_scale4_c3 = blk(in_d[2], self.mid_d, 1, 1, 0, 2)
        self.conv_scale5_c3 = blk(in_d[2], self.mid_d, 1, 1, 0, 4)
        self.conv_scale1_c4 = blk(in_d[3], self.mid_d, 1, 1, 0)
        self.conv_scale2_c4 = blk(in_d[3], self.mid_d, 1, 1, 0)
        self.conv_scale3_c4 = blk(in_d[3], self.mid_d, 1, 1, 0)
        self.conv_scale4_c4 = blk(in_d[3], self.mid_d, 3, 1, 1)
        self.conv_scale5_c4 = blk(in_d[3], self.mid_d, 1, 1, 0, 2)
        self.conv_scale1_c5 = blk(in_d[4], self.mid_d, 1, 1, 0)
        self.conv_scale2_c5 = blk(in_d[4], self.mid_d, 1, 1, 0)
        self.conv_scale3_c5 = blk(in_d[4], self.mid_d, 1, 1, 0)
        self.conv_scale4_c5 = blk(in_d[4], self.mid_d, 1, 1, 0)
        self.conv_scale5_c5 = blk(in_d[4], self.mid_d, 3, 1, 1)

        new_fuse_d = self.mid_d * 5 + self.out_d
        self.conv_aggregation_s1 = FeatureFusionModule_WithFreqResidual(new_fuse_d, self.in_d[0], self.out_d)
        self.conv_aggregation_s2 = FeatureFusionModule_WithFreqResidual(new_fuse_d, self.in_d[1], self.out_d)
        self.conv_aggregation_s3 = FeatureFusionModule_WithFreqResidual(new_fuse_d, self.in_d[2], self.out_d)
        self.conv_aggregation_s4 = FeatureFusionModule_WithFreqResidual(new_fuse_d, self.in_d[3], self.out_d)
        self.conv_aggregation_s5 = FeatureFusionModule_WithFreqResidual(new_fuse_d, self.in_d[4], self.out_d)

    def _process_one_stream(self, spatial_features, freq_features):
        c1, c2, c3, c4, c5 = spatial_features
        f1, f2, f3, f4, f5 = freq_features

        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)
        c1_s5 = self.conv_scale5_c1(c1)

        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=2, mode='bilinear', align_corners=False)
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)
        c2_s5 = self.conv_scale5_c2(c2)

        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=4, mode='bilinear', align_corners=False)
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=2, mode='bilinear', align_corners=False)
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)
        c3_s5 = self.conv_scale5_c3(c3)

        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=8, mode='bilinear', align_corners=False)
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=4, mode='bilinear', align_corners=False)
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=2, mode='bilinear', align_corners=False)
        c4_s4 = self.conv_scale4_c4(c4)
        c4_s5 = self.conv_scale5_c4(c4)

        c5_s1 = F.interpolate(self.conv_scale1_c5(c5), scale_factor=16, mode='bilinear', align_corners=False)
        c5_s2 = F.interpolate(self.conv_scale2_c5(c5), scale_factor=8, mode='bilinear', align_corners=False)
        c5_s3 = F.interpolate(self.conv_scale3_c5(c5), scale_factor=4, mode='bilinear', align_corners=False)
        c5_s4 = F.interpolate(self.conv_scale4_c5(c5), scale_factor=2, mode='bilinear', align_corners=False)
        c5_s5 = self.conv_scale5_c5(c5)

        s1 = self.conv_aggregation_s1(torch.cat([c1_s1, c2_s1, c3_s1, c4_s1, c5_s1, f1], dim=1), c1, f1)
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2, c2_s2, c3_s2, c4_s2, c5_s2, f2], dim=1), c2, f2)
        s3 = self.conv_aggregation_s3(torch.cat([c1_s3, c2_s3, c3_s3, c4_s3, c5_s3, f3], dim=1), c3, f3)
        s4 = self.conv_aggregation_s4(torch.cat([c1_s4, c2_s4, c3_s4, c4_s4, c5_s4, f4], dim=1), c4, f4)
        s5 = self.conv_aggregation_s5(torch.cat([c1_s5, c2_s5, c3_s5, c4_s5, c5_s5, f5], dim=1), c5, f5)

        return s1, s2, s3, s4, s5

    def forward(self, x1_features, x2_features):
        x1_freq, x2_freq = self.frequency_processor(x1_features, x2_features)
        x1_enhanced = self._process_one_stream(x1_features, x1_freq)
        x2_enhanced = self._process_one_stream(x2_features, x2_freq)
        return x1_enhanced, x2_enhanced