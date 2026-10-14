import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_m
from skimage.metrics import structural_similarity as ssim_m

# ==========================================
# 1. 核心评估指标 (最新修正版)
# ==========================================

def calculate_uciqe(img_bgr):
    """计算水下色彩质量指标 (UCIQE)"""
    if img_bgr is None or img_bgr.size == 0: return 0.0
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    chroma = np.sqrt(a.astype(np.float32)**2 + b.astype(np.float32)**2)
    return (0.4662 * np.std(chroma) + 0.2745 * (np.max(l) - np.min(l)) + 0.2593 * np.mean(chroma)) / 100.0

def calculate_eme(image, window_size):
    """计算 EME (Enhanced Measure of Enhancement)，UIQM 专用子函数"""
    H, W = image.shape
    step = window_size
    eme_sum = 0.0
    count = 0
    for i in range(0, H - step, step):
        for j in range(0, W - step, step):
            block = image[i:i+step, j:j+step]
            block_min = np.min(block) + 1e-6  # 防止除0
            block_max = np.max(block) + 1e-6
            if block_max > block_min:
                eme_sum += np.log(block_max / block_min)
                count += 1
    return eme_sum / count if count > 0 else 0.0

def calculate_uiqm(img_bgr):
    """计算水下图像质量测量指标 UIQM（论文标准实现）"""
    if img_bgr is None or img_bgr.size == 0: return 0.0
    img_float = img_bgr.astype(np.float32) / 255.0  # 归一化到 [0,1]
    # UICM 色彩分量
    R, G, B = img_float[:, :, 2], img_float[:, :, 1], img_float[:, :, 0]
    rg, yb = R - G, 0.5 * (R + G) - B
    uicm = -0.0268 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.1586 * np.sqrt(np.var(rg) + np.var(yb))
    # UISM 锐度分量
    uism = calculate_eme(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 8)
    # UIConM 对比度分量
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    uiconm = calculate_eme(lab[:, :, 0], 8)
    # 论文标准权重
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return round(float(uiqm), 4)

# ==========================================
# 2. 模型架构定义 (HUENet, FUnIEGAN, WaterNet)
# ==========================================
from models.huenet import HUENet

class GeneratorFunieGAN(nn.Module):
    def __init__(self):
        super().__init__()
        def down(i, o, bn=True):
            l = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
            if bn: l.append(nn.BatchNorm2d(o, momentum=0.8))
            l.append(nn.LeakyReLU(0.2)); return nn.Sequential(*l)
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o, momentum=0.8), nn.ReLU())
        self.d1, self.d2, self.d3, self.d4, self.d5 = down(3,32,False), down(32,128), down(128,256), down(256,256), down(256,256,False)
        self.u1, self.u2, self.u3, self.u4 = up(256,256), up(512,256), up(512,128), up(256,32)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(64,3,4,padding=1), nn.Tanh())
    def forward(self, x):
        x1=self.d1(x); x2=self.d2(x1); x3=self.d3(x2); x4=self.d4(x3); x5=self.d5(x4)
        o1=self.u1(x5); o1=torch.cat([o1,x4],1); o2=self.u2(o1); o2=torch.cat([o2,x3],1)
        o3=self.u3(o2); o3=torch.cat([o3,x2],1); o4=self.u4(o3); o4=torch.cat([o4,x1],1)
        return self.final(o4)

class WaterNet_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(1, s))
        self.l1, self.l2, self.l3, self.l5, self.l6 = block(3,16,1), block(16,32,1), block(32,64,2), block(64,128,2), block(128,128,2)
        self.l10 = nn.Conv2d(128, 3, 1, 1)
    def forward(self, x):
        x = x * 2.0 - 1.0 # 量程对齐
        o = self.l6(self.l5(self.l3(self.l2(self.l1(x)))))
        return torch.sigmoid(F.interpolate(self.l10(o), size=(256, 256), mode='bilinear'))

# ==========================================
# 3. 对比实验主程序
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_root = "./comparison_results"
    methods = ["CLAHE", "FUnIEGAN", "WaterNet", "HUENet"]
    for m in methods + ["Raw", "GT"]: os.makedirs(os.path.join(save_root, m), exist_ok=True)

    # 加载模型
    def load(m, p, key=None):
        if not os.path.exists(p): return m.to(device).eval()
        d = torch.load(p, map_location=device, weights_only=False)
        if key and key in d: d = d[key]
        new_d = {k.replace('module.', ''): v for k, v in d.items()}
        m.load_state_dict(new_d, strict=False); return m.to(device).eval()

    m_h = load(HUENet(), 'checkpoints/best_model.pth', 'model')
    m_f = load(GeneratorFunieGAN(), './models/funie_generator.pth')
    m_w = load(WaterNet_VGG(), './model_weight.pth')

    raw_dir, ref_dir = "./data/UIEB/raw-890", "./data/UIEB/reference-890"
    test_names = sorted(os.listdir(raw_dir))[800:]
    tf_01 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    tf_norm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    stats = {m: {"psnr":[], "ssim":[], "uciqe":[], "uiqm":[]} for m in methods}

    print("🚀 启动跨模型指标对比实验...")
    for name in tqdm(test_names):
        raw_pil = Image.open(os.path.join(raw_dir, name)).convert('RGB')
        gt_np = np.array(Image.open(os.path.join(ref_dir, name)).convert('RGB').resize((256,256))).astype(np.float32)/255.0

        with torch.no_grad():
            # 推理
            out_h = m_h(tf_01(raw_pil).unsqueeze(0).to(device)).squeeze().cpu().permute(1,2,0).numpy().clip(0,1)
            out_f = ((m_f(tf_norm(raw_pil).unsqueeze(0).to(device)).squeeze().cpu().permute(1,2,0).numpy() + 1.0)/2.0).clip(0,1)
            out_w = m_w(tf_01(raw_pil).unsqueeze(0).to(device)).squeeze().cpu().permute(1,2,0).numpy().clip(0,1)
            # CLAHE
            bgr = cv2.cvtColor(np.array(raw_pil.resize((256,256))), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l,a,b_ = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            out_c = cv2.cvtColor(cv2.cvtColor(cv2.merge((l,a,b_)), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)/255.0

        # 记录
        for m_name, m_out in zip(methods, [out_c, out_f, out_w, out_h]):
            bgr_final = cv2.cvtColor((m_out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_root, m_name, name), bgr_final)
            stats[m_name]["psnr"].append(psnr_m(gt_np, m_out, data_range=1.0))
            stats[m_name]["ssim"].append(ssim_m(gt_np, m_out, data_range=1.0, channel_axis=2))
            stats[m_name]["uciqe"].append(calculate_uciqe(bgr_final))
            stats[m_name]["uiqm"].append(calculate_uiqm(bgr_final))

    print("\n" + "="*75)
    print(f"{'Method':<15} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'UCIQE ↑':<8} | {'UIQM ↑':<8}")
    print("-" * 75)
    for m in methods:
        s = stats[m]
        print(f"{m:<15} | {np.mean(s['psnr']):.4f} | {np.mean(s['ssim']):.4f} | {np.mean(s['uciqe']):.4f} | {np.mean(s['uiqm']):.4f}")
    print("="*75)

if __name__ == "__main__": main()