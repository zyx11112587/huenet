import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import CombinedDataset
from models.huenet import HUENet
from pytorch_msssim import SSIM
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio

def train():
    # 1. 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # 数据加载
    uieb_path = "./huenet/data/UIEB"
    euvp_path = "./huenet/data/EUVP"
    train_loader = DataLoader(CombinedDataset(uieb_path, euvp_path, 'train', tf), batch_size=8, shuffle=True)
    val_loader = DataLoader(CombinedDataset(uieb_path, None, 'val', tf), batch_size=1)

    # 2. 初始化模型、优化器和损失函数
    model = HUENet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    
    # 3. --- 断点续传逻辑 ---
    checkpoint_path = 'checkpoints/latest_model.pth'
    start_epoch = 0
    best_psnr = 0.0

    if os.path.exists(checkpoint_path):
        print(f"--- 发现断点文件: {checkpoint_path}，正在恢复训练... ---")
        # map_location 确保在不同硬件环境也能加载
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model'])
        
        # 加载优化器状态 (保证学习率等状态不丢失)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载轮数和历史最高分
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_psnr' in checkpoint:
            best_psnr = checkpoint['best_psnr']
            
        print(f"--- 已恢复！从第 {start_epoch + 1} 轮开始训练，当前历史最高 PSNR: {best_psnr:.2f} ---")
    else:
        print("--- 未发现断点，从第 1 轮开始训练 ---")

    # 4. 训练循环
    epochs = 100
    for epoch in range(start_epoch, epochs):
        model.train()
        t_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for img_A, img_B in t_bar:
            img_A, img_B = img_A.to(device), img_B.to(device)
            optimizer.zero_grad()
            out = model(img_A)
            
            # 复合损失计算
            loss = 0.5 * torch.nn.functional.l1_loss(out, img_B) + 0.5 * (1 - ssim_module(out, img_B))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            t_bar.set_postfix(loss=loss.item())

        # 5. 验证环节
        model.eval()
        psnrs = []
        with torch.no_grad():
            for i, (img_A, img_B) in enumerate(val_loader):
                out = model(img_A.to(device)).squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
                gt = img_B.squeeze().numpy().transpose(1, 2, 0)
                psnrs.append(peak_signal_noise_ratio(gt, out, data_range=1.0))
        
        avg_psnr = np.mean(psnrs)
        print(f"Epoch {epoch+1} 结束 | PSNR: {avg_psnr:.2f}")

        # 6. --- 保存状态字典 ---
        # 封装所有必要信息
        checkpoint_state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': max(avg_psnr, best_psnr)
        }
        
        # 保存为最新模型 (供恢复使用)
        torch.save(checkpoint_state, 'checkpoints/latest_model.pth')
        
        # 如果当前性能最好，保存为 best_model.pth (供演示系统和最终论文使用)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(checkpoint_state, 'checkpoints/best_model.pth')
            print(f"🌟 发现更好的模型！Best PSNR 已更新为: {best_psnr:.2f}")

if __name__ == "__main__":
    train()