import torch
import torch.nn as nn
from pytorch_msssim import SSIM

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, pred, target):
        # 色彩恒常性损失 (防止偏色)
        mu_p = torch.mean(pred, dim=(2,3))
        col_loss = torch.mean(torch.pow(mu_p[:,0]-mu_p[:,1], 2) + torch.pow(mu_p[:,1]-mu_p[:,2], 2))
        return 0.4 * self.l1(pred, target) + 0.5 * (1 - self.ssim(pred, target)) + 0.1 * col_loss