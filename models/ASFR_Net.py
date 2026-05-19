import torch
import torch.nn as nn
import torch.nn.functional as F

from .MobileNetV2 import mobilenet_v2
from .SFEM import SpatioFrequencyReinforcementModule
from .TFM import TemporalFusionModule_five
from .Decoder import SimplifiedDecoder

class ASFR_Net(nn.Module):
    """
    ASFR-Net 核心网络架构 (原 BaseNet)。
    保持变量名不变，以完美兼容旧的预训练权重。
    """
    def __init__(self, input_nc=3, output_nc=1):
        super(ASFR_Net, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        channels = [16, 24, 32, 96, 320]

        self.en_d = 32
        self.mid_d = self.en_d * 2

        # 核心模块: 保持 swa, tfm, decoder 命名不变
        self.swa = SpatioFrequencyReinforcementModule(channels, self.mid_d)
        self.tfm = TemporalFusionModule_five(self.mid_d, self.en_d * 2)
        self.decoder = SimplifiedDecoder(self.mid_d)

    def forward(self, x1, x2):
        # 1. 骨干网络特征提取
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        x1_features = [x1_1, x1_2, x1_3, x1_4, x1_5]
        x2_features = [x2_1, x2_2, x2_3, x2_4, x2_5]

        # 2. 时空-频率特征聚合 (SFEM)
        [x1_1, x1_2, x1_3, x1_4, x1_5], [x2_1, x2_2, x2_3, x2_4, x2_5] = self.swa(
            [x1_1, x1_2, x1_3, x1_4, x1_5], [x2_1, x2_2, x2_3, x2_4, x2_5]
        )

        # 3. 时序特征差分与融合 (TFM)
        c1, c2, c3, c4, c5 = self.tfm(x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5)

        # 4. 级联解码器
        mask_p1, mask_p2, mask_p3, mask_p4 = self.decoder(c1, c2, c3, c4, c5)

        # 5. 上采样至原图分辨率
        mask_p1 = torch.sigmoid(F.interpolate(mask_p1, scale_factor=2, mode='bilinear', align_corners=False))
        mask_p2 = torch.sigmoid(F.interpolate(mask_p2, scale_factor=4, mode='bilinear', align_corners=False))
        mask_p3 = torch.sigmoid(F.interpolate(mask_p3, scale_factor=8, mode='bilinear', align_corners=False))
        mask_p4 = torch.sigmoid(F.interpolate(mask_p4, scale_factor=16, mode='bilinear', align_corners=False))

        return mask_p1, mask_p2, mask_p3, mask_p4, x1_features, x2_features