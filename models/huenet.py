import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x): 
        return torch.relu(x + self.conv(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x):
        B, C, H, W = x.shape
        f = x.flatten(2).transpose(1, 2)
        f = f + self.attn(self.norm(f), self.norm(f), self.norm(f))[0]
        f = f + self.mlp(self.norm(f))
        return f.transpose(1, 2).reshape(B, C, H, W)

# class HUENet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder 保持不变
#         self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(True), ResBlock(32))
#         self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True), ResBlock(64))
#         self.bottleneck = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, 1, 1), TransformerBlock(128))
        
#         # Decoder 增加通道对齐
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.dec2 = nn.Sequential(nn.Conv2d(128+64, 64, 3, 1, 1), nn.ReLU(True))
        
#         self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.dec1 = nn.Sequential(nn.Conv2d(64+32, 32, 3, 1, 1), nn.ReLU(True))
        
#         # 最终输出层：32通道 -> 3通道，加上 Sigmoid 保证量程 [0, 1]
#         self.final = nn.Sequential(
#             nn.Conv2d(32, 3, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         s1 = self.enc1(x)              # [B, 32, 256, 256]
#         s2 = self.enc2(s1)             # [B, 64, 128, 128]
#         b = self.bottleneck(s2)        # [B, 128, 64, 64]
        
#         d2 = self.up2(b)               # [B, 128, 128, 128]
#         d2 = self.dec2(torch.cat([d2, s2], 1)) # [B, 64, 128, 128]
        
#         d1 = self.up1(d2)              # [B, 64, 256, 256]
#         d1 = self.dec1(torch.cat([d1, s1], 1)) # [B, 32, 256, 256]
        
#         # 【关键修复】调用 final 层，返回 3 通道图像
#         return self.final(d1)
class HUENet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- 1. 浅层特征提取 (增加初步卷积层) ---
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- 2. Encoder (保持残差结构) ---
        self.enc1 = nn.Sequential(ResBlock(32), ResBlock(32)) # 增加深度
        self.down1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True), ResBlock(64))
        self.down2 = nn.MaxPool2d(2)

        # --- 3. Bottleneck (Transformer) ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            TransformerBlock(128)
        )
        
        # --- 4. Decoder (强化融合深度：使用 ResBlock 代替单层 Conv) ---
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2_fuse = nn.Sequential(
            nn.Conv2d(128+64, 64, 1, 1, 0), # 1x1 卷积先降维
            ResBlock(64),                  # 深度消化特征
            ResBlock(64)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_fuse = nn.Sequential(
            nn.Conv2d(64+32, 32, 1, 1, 0), # 1x1 卷积先降维
            ResBlock(32),
            ResBlock(32)
        )
        
        # --- 5. 精细化阶段 (Refinement Layer) ---
        # 在最终输出前，增加两层卷积处理，进一步平滑色彩
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial
        x_init = self.init_conv(x)
        
        # Encoder
        s1 = self.enc1(x_init)         
        s2 = self.enc2(self.down1(s1)) 
        
        # Bottleneck
        b = self.bottleneck(self.down2(s2)) 
        
        # Decoder with deeper fusion
        d2 = self.up2(b)
        d2 = self.dec2_fuse(torch.cat([d2, s2], 1)) 
        
        d1 = self.up1(d2)
        d1 = self.dec1_fuse(torch.cat([d1, s1], 1))
        
        # Refinement
        out = self.refine(d1)
        
        return self.final(out)