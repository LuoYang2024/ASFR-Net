import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image

# 1. 自动获取项目根目录，解决 ModuleNotFoundError
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.ASFR_Net import ASFR_Net
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter


@torch.no_grad()
def Validate(args):
    # 2. 实例化模型并打印参数量 (校验是否为 4472502)
    model = ASFR_Net().cuda()
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"[*] 模型加载成功。总参数量: {total_params}")

    # 3. 加载权重
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"找不到权重文件: {args.weight}")
    print(f"[*] 正在加载权重: {args.weight}")
    model.load_state_dict(torch.load(args.weight, map_location='cuda'))
    model.eval()

    # 4. 关键修正：归一化参数必须与 train_2.py 严格一致
    # 原版 train_2.py 使用的是 ImageNet 的均值和方差
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    val_trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.PrepareForTensorWithFlag(),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=val_trans)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print(f"[*] 测试集样本数: {len(test_data)}")
    meter = ConfuseMatrixMeter(n_class=2)

    for iter_id, batched_inputs in enumerate(testLoader):
        img, target, _, _, _ = batched_inputs
        pre_img = img[:, 0:3].cuda()
        post_img = img[:, 3:6].cuda()

        # 5. 推理 (只取最高分辨率的预测图 out1)
        out1, out2, out3, out4, _, _ = model(pre_img, post_img)

        # 二值化处理
        pred = torch.where(out1 > 0.5, torch.ones_like(out1), torch.zeros_like(out1)).long()

        # 更新混淆矩阵
        meter.update_cm(pr=pred.cpu().numpy(), gt=target.numpy())

        if iter_id % 100 == 0:
            print(f"  -> 已处理: [{iter_id}/{len(testLoader)}]")

    # 6. 打印结果
    scores = meter.get_scores()
    print("\n" + "=" * 50)
    print(f"[Final Test Result on {args.file_root}]")
    print(f"  F1-Score  : {scores['F1'] * 100:.2f} %")
    print(f"  IoU       : {scores['IoU'] * 100:.2f} %")
    print(f"  Precision : {scores['precision'] * 100:.2f} %")
    print(f"  Recall    : {scores['recall'] * 100:.2f} %")
    print(f"  Overall Acc: {scores['OA'] * 100:.2f} %")
    print("=" * 50)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default=r"E:\MCD", help='Dataset path')
    parser.add_argument('--weight', default=r'E:\ASFR-Net\tools\results_SR_iter_40000_lr_0.0005\best_model_G.pth')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    args = parser.parse_args()
    Validate(args)