# tools/train.py
import sys
import os
import time
import numpy as np
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset

# 自动获取项目根目录，解决导入错误
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.ASFR_Net import ASFR_Net
from models.Discriminator import AdversarialNetwork, BoundedChangeContrastiveLoss, calc_coeff
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter


# ==============================================================================
# 1. 数据增强与包装器
# ==============================================================================
def rand_bbox(size, lam, scale_range=(0.5, 1.5)):
    H, W = size[1], size[2]
    cut_rat = np.sqrt(1. - lam)
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    cut_w, cut_h = int(W * cut_rat * scale_factor), int(H * cut_rat * scale_factor)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H), \
        np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)


class CutMixDataset(Dataset):
    def __init__(self, dataset, beta=1.0, prob=0.5, scale_range=(0.5, 1.5)):
        self.dataset = dataset
        self.beta = beta
        self.prob = prob
        self.scale_range = scale_range

    def __getitem__(self, index):
        img1, target1, pre_mask1, post_mask1, flag1 = self.dataset[index]
        if self.beta > 0 and random.random() < self.prob:
            rand_index = random.randint(0, len(self.dataset) - 1)
            img2, target2, _, _, _ = self.dataset[rand_index]
            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = rand_bbox(img1.size(), lam, self.scale_range)
            img1[:, x1:x2, y1:y2] = img2[:, x1:x2, y1:y2]
            target1[:, x1:x2, y1:y2] = target2[:, x1:x2, y1:y2]
        return img1, target1, pre_mask1, post_mask1, flag1

    def __len__(self):
        return len(self.dataset)


# ==============================================================================
# 2. 损失函数与策略
# ==============================================================================
def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    dice = (2 * inter + 1e-5) / (inputs.sum() + targets.sum() + 1e-5)
    return bce + 1 - dice


def improved_training_strategy(epoch):
    if epoch < 5:
        return 0.0, 0.0
    elif epoch < 15:
        p = (epoch - 5) / 10.0
        return 0.001 * (1 / (1 + np.exp(-10 * (p - 0.5)))), 0.002 * (1 / (1 + np.exp(-10 * (p - 0.5))))
    return 0.001, 0.002


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1.0):
    lr = args.lr * (1 - iter / (max_batches * args.max_epochs)) ** 0.9
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ==============================================================================
# 3. 核心训练与验证逻辑
# ==============================================================================
def train(args, train_loader, model_G, optimizer_G, model_D, optimizer_D, epoch, max_batches, cur_iter):
    model_G.train()
    if model_D: model_D.train()
    contrastive_loss_fn = BoundedChangeContrastiveLoss(alpha=1.0, beta=0.05).cuda()
    meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    l_adv, l_con = improved_training_strategy(epoch)
    if not model_D: l_adv = 0.0

    for iter, (img, target, _, _, flag) in enumerate(train_loader):
        pre_img, post_img, target_var = img[:, 0:3].cuda(), img[:, 3:6].cuda(), target.float().cuda()
        flag = flag.cuda()
        lr_G = adjust_learning_rate(args, optimizer_G, epoch, iter + cur_iter, max_batches)
        if optimizer_D: adjust_learning_rate(args, optimizer_D, epoch, iter + cur_iter, max_batches, 0.2)

        optimizer_G.zero_grad()
        out1, out2, out3, out4, x1_f, x2_f = model_G(pre_img, post_img)
        loss_cd = sum([BCEDiceLoss(o, target_var) for o in [out1, out2, out3, out4]])
        loss_G_adv, loss_con = 0, 0

        if l_adv > 0:
            prob = torch.softmax(torch.cat([1 - torch.sigmoid(out1), torch.sigmoid(out1)], dim=1), dim=1).detach()
            coeff = calc_coeff(cur_iter + iter, max_iter=args.max_steps)
            for i in range(3):
                feat_to_fool = torch.where(flag.view(-1, 1, 1, 1) == 1, x1_f[i], x2_f[i])
                cond_feat = torch.einsum('bchw,bkhw->bckhw', feat_to_fool,
                                         F.adaptive_max_pool2d(prob, feat_to_fool.shape[2:])).flatten(1, 2)
                loss_G_adv += -torch.mean(model_D(cond_feat, coeff, i))
                loss_con += contrastive_loss_fn(x1_f[i], x2_f[i], target_var)

        loss_G_total = loss_cd + loss_G_adv * l_adv + loss_con * l_con
        loss_G_total.backward()
        optimizer_G.step()

        loss_D_total = 0
        if l_adv > 0:
            optimizer_D.zero_grad()
            for i in range(3):
                real_f = torch.where(flag.view(-1, 1, 1, 1) == 1, x2_f[i].detach(), x1_f[i].detach())
                fake_f = torch.where(flag.view(-1, 1, 1, 1) == 1, x1_f[i].detach(), x2_f[i].detach())
                p_r = F.adaptive_max_pool2d(prob, real_f.shape[2:])
                c_real = torch.einsum('bchw,bkhw->bckhw', real_f, p_r).flatten(1, 2).requires_grad_(True)
                c_fake = torch.einsum('bchw,bkhw->bckhw', fake_f, p_r).flatten(1, 2).detach()
                out_r, out_f = model_D(c_real, 0.0, i), model_D(c_fake, 0.0, i)
                loss_D = torch.mean(F.relu(1. - out_r)) + torch.mean(F.relu(1. + out_f))
                grad_r = torch.autograd.grad(outputs=out_r.sum(), inputs=c_real, create_graph=True, retain_graph=True)[
                    0]
                loss_D += 5.0 * (grad_r.view(grad_r.size(0), -1).norm(2, dim=1) ** 2).mean()
                loss_D_total += loss_D
            loss_D_total.backward()
            optimizer_D.step()

        epoch_loss.append(loss_G_total.item())
        pred = torch.where(out1 > 0.5, torch.ones_like(out1), torch.zeros_like(out1)).long()
        f1 = meter.update_cm(pred.cpu().numpy(), target_var.cpu().numpy())

        if iter % 10 == 0:
            d_val = loss_D_total.item() if isinstance(loss_D_total, torch.Tensor) else loss_D_total
            print(
                f"\rE[{epoch}/{args.max_epochs - 1}] I[{iter}/{max_batches}] | F1:{f1:.4f} | GLoss:{loss_G_total.item():.3f} | DLoss:{d_val:.3f}",
                end="")

    return sum(epoch_loss) / len(epoch_loss), meter.get_scores(), lr_G


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    for img, target, _, _, _ in val_loader:
        out1, _, _, _, _, _ = model(img[:, 0:3].cuda(), img[:, 3:6].cuda())
        pred = torch.where(out1 > 0.5, torch.ones_like(out1), torch.zeros_like(out1)).long()
        meter.update_cm(pred.cpu().numpy(), target.numpy())
    return meter.get_scores()


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
def trainValidateSegmentation(args):
    # 保存原始数据集名称用于目录命名
    dataset_name = args.file_root

    # 路径转换映射 (与最初代码保持一致)
    if args.file_root == 'LEVIR':
        args.file_root = r'E:\Datasets\LEVIR-CD-256'  # 修改为你的实际路径
    elif args.file_root == 'BCDD':
        args.file_root = r'E:\Datasets\BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = r'E:\Datasets\SYSU'
    elif args.file_root == 'MCD':
        args.file_root = r'E:\MCD'
    elif args.file_root == 'VisNIR_HCD':
        args.file_root = r'E:\VisNIR_HCD'

    # 设置保存目录 (防止冒号冲突)
    args.savedir = f"{args.savedir}_{dataset_name}_iter_{args.max_steps}_lr_{args.lr}"
    os.makedirs(args.savedir, exist_ok=True)

    model_G = ASFR_Net().cuda()
    model_D = AdversarialNetwork([16, 24, 32]).cuda() if args.adversarial else None

    # 归一化参数对齐
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    train_trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])
    val_trans = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.PrepareForTensorWithFlag(),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=train_trans)
    if args.cutmix: train_data = CutMixDataset(train_data, beta=1.0, prob=0.6)
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, drop_last=True)
    valLoader = torch.utils.data.DataLoader(myDataLoader.Dataset("val", file_root=args.file_root, transform=val_trans),
                                            batch_size=args.batch_size)
    testLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset("test", file_root=args.file_root, transform=val_trans), batch_size=1)

    opt_G = torch.optim.AdamW(model_G.parameters(), args.lr, weight_decay=1e-2)
    opt_D = torch.optim.AdamW(model_D.parameters(), args.lr * 0.2, weight_decay=1e-2) if model_D else None

    # [断点恢复逻辑]
    start_epoch = 0
    ckpt_path = os.path.join(args.savedir, 'checkpoint.pth.tar')
    if args.resume and os.path.exists(ckpt_path):
        print(f"[*] 发现检查点，正在恢复训练: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        model_G.load_state_dict(ckpt['state_dict_G'])
        if model_D and ckpt.get('state_dict_D'): model_D.load_state_dict(ckpt['state_dict_D'])
        if 'optimizer_G' in ckpt: opt_G.load_state_dict(ckpt['optimizer_G'])

    args.max_epochs = int(np.ceil(args.max_steps / len(trainLoader)))
    log_file = open(os.path.join(args.savedir, args.logFile), 'a' if args.resume else 'w')
    if not args.resume:
        log_file.write(f"Parameters: {sum([np.prod(p.size()) for p in model_G.parameters()])}\n")
        log_file.write("Epoch\tKappa(val)\tIoU(val)\tF1(train)\tF1(val)\tR(val)\tP(val)\n")

    max_f1_val = 0
    for epoch in range(start_epoch, args.max_epochs):
        loss_tr, score_tr, lr_G = train(args, trainLoader, model_G, opt_G, model_D, opt_D, epoch, len(trainLoader),
                                        epoch * len(trainLoader))
        print()

        if epoch == 0: continue
        score_va = val(args, valLoader, model_G)

        log_file.write(
            f"{epoch}\t{score_va['Kappa']:.4f}\t{score_va['IoU']:.4f}\t{score_tr['F1']:.4f}\t{score_va['F1']:.4f}\t{score_va['recall']:.4f}\t{score_va['precision']:.4f}\n")
        log_file.flush()

        # 保存最新进度
        torch.save({
            'epoch': epoch + 1,
            'state_dict_G': model_G.state_dict(),
            'state_dict_D': model_D.state_dict() if model_D else None,
            'optimizer_G': opt_G.state_dict(),
        }, ckpt_path)

        if score_va['F1'] > max_f1_val:
            max_f1_val = score_va['F1']
            torch.save(model_G.state_dict(), os.path.join(args.savedir, 'best_model_G.pth'))
            print(f"[*] Best model saved: Epoch {epoch}, Val F1: {max_f1_val:.4f}")

    # 训练结束自动执行测试
    print("\n" + "=" * 20 + " 自动测试开始 " + "=" * 20)
    best_model_path = os.path.join(args.savedir, 'best_model_G.pth')
    if os.path.exists(best_model_path):
        model_G.load_state_dict(torch.load(best_model_path))
    model_G.eval()
    score_te = val(args, testLoader, model_G)
    print(f"\n[Test Result on {dataset_name}] F1: {score_te['F1']:.4f} | IoU: {score_te['IoU']:.4f}")
    log_file.write(f"\n[Final Test]\tF1:{score_te['F1']:.4f}\tIoU:{score_te['IoU']:.4f}\tOA:{score_te['OA']:.4f}\n")
    log_file.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="VisNIR_HCD", help='Dataset name: LEVIR | BCDD | SYSU | MCD')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--adversarial', default=True, type=bool)
    parser.add_argument('--cutmix', default=True, type=bool)
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--savedir', default='./results')
    parser.add_argument('--logFile', default='trainValLog.txt')
    parser.add_argument('--onGPU', default=True, type=bool)
    parser.add_argument('--lr_mode', default='poly')
    args = parser.parse_args()
    trainValidateSegmentation(args)