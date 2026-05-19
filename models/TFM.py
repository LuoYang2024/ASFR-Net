import torch
import torch.nn as nn

class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2_f(self.relu(self.conv_branch2(x) + x_branch1))
        x_branch3 = self.conv_branch3_f(self.relu(self.conv_branch3(x) + x_branch2))
        x_branch4 = self.conv_branch4_f(self.relu(self.conv_branch4(x) + x_branch3))
        return self.relu(self.conv_branch5(x) + x_branch4)


class TemporalFusionModule_five(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(TemporalFusionModule_five, self).__init__()
        self.tffm_x1 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x2 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x4 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x5 = TemporalFeatureFusionModule(in_d, out_d)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5):
        c1 = self.tffm_x1(x1_1, x2_1)
        c2 = self.tffm_x2(x1_2, x2_2)
        c3 = self.tffm_x3(x1_3, x2_3)
        c4 = self.tffm_x4(x1_4, x2_4)
        c5 = self.tffm_x5(x1_5, x2_5)
        return c1, c2, c3, c4, c5