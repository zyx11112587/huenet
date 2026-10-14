
import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.huenet import HUENet
from skimage.metrics import peak_signal_noise_ratio as psnr_m
from skimage.metrics import structural_similarity as ssim_m
from tqdm import tqdm

# --- 学术指标：UCIQE 计算函数 ---
def calculate_uciqe(img_bgr):
    """计算水下色彩质量指标 (UCIQE)"""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    chroma = np.sqrt(a.astype(np.float32)**2 + b.astype(np.float32)**2)
    sigma_c = np.std(chroma)
    con_l = np.max(l) - np.min(l)
    mu_c = np.mean(chroma)
    uciqe = 0.4662 * sigma_c + 0.2745 * con_l + 0.2593 * mu_c
    return uciqe / 100.0

# --- 学术指标：UIQM 计算相关函数 ---
def calculate_eme(img, block_size=8):
    """计算增强评价准则 (EME)，用于 UIQM 的锐度和对比度分量"""
    m, n = img.shape
    num_m = m // block_size
    num_n = n // block_size
    eme = 0
    for i in range(num_m):
        for j in range(num_n):
            block = img[i*block_size : (i+1)*block_size, j*block_size : (j+1)*block_size]
            max_val = np.max(block)
            min_val = np.min(block)
            if min_val > 0 and max_val > 0:
                eme += 20 * np.log(max_val / min_val + 1e-5)
    return eme / (num_m * num_n + 1e-5)

def calculate_uiqm(img_bgr):
    """计算水下图像质量测量指标 (UIQM)"""
    img_bgr = img_bgr.astype(np.float32)
    # 1. UICM (色彩分量)
    R, G, B = img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]
    rg = R - G
    yb = 0.5 * (R + G) - B
    uicm = -0.0268 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.1586 * np.sqrt(np.var(rg) + np.var(yb))

    # 2. UISM (锐度分量)
    gray = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    uism = calculate_eme(gray, 8)

    # 3. UIConM (对比度分量)
    lab = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    uiconm = calculate_eme(l_channel, 8)

    # UIQM = c1*UICM + c2*UISM + c3*UIConM (经典系数)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm

def test():
    # 1. 硬件与路径配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_model.pth'
    raw_dir = "./data/UIEB/raw-890"
    ref_dir = "./data/UIEB/reference-890"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 加载模型
    model = HUENet().to(device)
    if not os.path.exists(model_path):
        model_path = 'checkpoints/latest_model.pth'
    
    print(f"--- 正在加载模型: {model_path} ---")
    # 适配 PyTorch 2.6+ 权重加载安全性要求
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except:
        ckpt = torch.load(model_path, map_location=device)
        
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 3. 准备测试数据 (UIEB 最后 90 张)
    test_imgs = sorted(os.listdir(raw_dir))[800:]
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    psnrs, ssims, uciqes, uiqms = [], [], [], []

    print(f"开始测试，共 {len(test_imgs)} 张图片...")
    with torch.no_grad():
        for n in tqdm(test_imgs):
            # 加载并Resize真值图用于全参考指标计算
            gt_pil = Image.open(os.path.join(ref_dir, n)).convert('RGB')
            gt_np = np.array(gt_pil.resize((256, 256))).astype(np.float32) / 255.0

            # 推理
            input_pil = Image.open(os.path.join(raw_dir, n)).convert('RGB')
            input_tensor = tf(input_pil).unsqueeze(0).to(device)
            out_t = model(input_tensor)
            
            # 转为 Numpy [0, 1] 范围
            out_np = out_t.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            # --- 计算指标 ---
            # 1. PSNR & SSIM
            psnrs.append(psnr_m(gt_np, out_np, data_range=1.0))
            ssims.append(ssim_m(gt_np, out_np, data_range=1.0, channel_axis=2))

            # 转为 BGR uint8 格式用于无参考指标计算和保存
            res_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 2. UCIQE
            uciqes.append(calculate_uciqe(res_bgr))
            
            # 3. UIQM
            uiqms.append(calculate_uiqm(res_bgr))

            # 保存图片
            cv2.imwrite(os.path.join(save_dir, n), res_bgr)

    # 4. 打印最终结果
    print("\n" + "="*45)
    print("       HUENet 毕业设计全量实验结果单")
    print("-" * 45)
    print(f"测试样本数: {len(test_imgs)}")
    print(f"平均 PSNR:  {np.mean(psnrs):.4f} dB (越大越好)")
    print(f"平均 SSIM:  {np.mean(ssims):.4f}    (越接近1越好)")
    print(f"平均 UCIQE: {np.mean(uciqes):.4f}    (色彩平衡，越大越好)")
    print(f"平均 UIQM:  {np.mean(uiqms):.4f}    (综合感官，越大越好)")
    print("-" * 45)
    print(f"增强图片已保存至: {os.path.join(os.getcwd(), save_dir)}")
    print("="*45)

if __name__ == "__main__":
    test()