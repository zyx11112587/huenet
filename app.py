import streamlit as st
import torch
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from torchvision import transforms
from models.huenet import HUENet
from skimage.metrics import peak_signal_noise_ratio as psnr_m
from skimage.metrics import structural_similarity as ssim_m
import time

# --- 1. 水下专业评价指标实现 ---

def calculate_uciqe(img_bgr):
    """计算水下色彩质量指标 (UCIQE)"""
    if img_bgr is None: return 0.0
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    chroma = np.sqrt(a.astype(np.float32)**2 + b.astype(np.float32)**2)
    sigma_c = np.std(chroma)
    con_l = np.max(l) - np.min(l)
    mu_c = np.mean(chroma)
    # 标准加权系数
    return (0.4662 * sigma_c + 0.2745 * con_l + 0.2593 * mu_c) / 100.0

# def calculate_eme(img, block_size=8):
#     """计算增强评价准则 (EME) 用于 UIQM 分量"""
#     m, n = img.shape
#     num_m, num_n = m // block_size, n // block_size
#     eme = 0
#     for i in range(num_m):
#         for j in range(num_n):
#             block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
#             max_v, min_v = np.max(block), np.min(block)
#             if min_v > 0 and max_v > 0:
#                 eme += 20 * np.log(max_v / min_v + 1e-5)
#     return eme / (num_m * num_n + 1e-5)

# def calculate_uiqm(img_bgr):
#     """计算水下图像质量测量指标 (UIQM)"""
#     if img_bgr is None: return 0.0
#     img_bgr = img_bgr.astype(np.float32)
#     # UICM (色彩分量)
#     R, G, B = img_bgr[:,:,2], img_bgr[:,:,1], img_bgr[:,:,0]
#     rg, yb = R - G, 0.5 * (R + G) - B
#     uicm = -0.0268 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.1586 * np.sqrt(np.var(rg) + np.var(yb))
#     # UISM (锐度分量)
#     gray = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
#     uism = calculate_eme(gray, 8)
#     # UIConM (对比度分量)
#     lab = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
#     uiconm = calculate_eme(lab[:,:,0], 8)
#     # 加权求和
#     return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm

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
    if count == 0:
        return 0.0
    return eme_sum / count

def calculate_uiqm(img_bgr):
    """
    计算水下图像质量测量指标 UIQM（论文标准实现）
    :param img_bgr: OpenCV 读取的 BGR 图像，uint8 格式
    :return: UIQM 分数
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    # 统一数据类型
    img_float = img_bgr.astype(np.float32) / 255.0  # 归一化到 [0,1]

    # ------------------- UICM 色彩分量 -------------------
    R = img_float[:, :, 2]
    G = img_float[:, :, 1]
    B = img_float[:, :, 0]

    rg = R - G
    yb = 0.5 * (R + G) - B

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    var_rg = np.var(rg)
    var_yb = np.var(yb)

    uicm = -0.0268 * np.sqrt(mean_rg ** 2 + mean_yb ** 2) + 0.1586 * np.sqrt(var_rg + var_yb)

    # ------------------- UISM 锐度分量 -------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    uism = calculate_eme(gray, 8)

    # ------------------- UIConM 对比度分量 -------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    uiconm = calculate_eme(l_channel, 8)

    # ------------------- 论文标准权重 -------------------
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm

    return round(float(uiqm), 4)
# --- 2. 系统核心功能设计 ---

st.set_page_config(page_title="HUENet水下图像增强系统", layout="wide", initial_sidebar_state="expanded")

# 侧边栏：硬件与模型状态
st.sidebar.title("🚀 系统控制面板")
st.sidebar.markdown("---")
device_info = "GPU (NVIDIA A10)" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"当前运行设备: **{device_info}**")

model_select = st.sidebar.selectbox("选择权重文件", ["best_model.pth", "latest_model.pth"])
model_dir = "./checkpoints"
dataset_root = st.sidebar.text_input("数据集根路径", "./huenet/data/UIEB")

@st.cache_resource
def get_model(name):
    path = os.path.join(model_dir, name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = HUENet().to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        net.eval()
        return net, device
    return None, None

model, dev = get_model(model_select)

# 主界面
st.title("🌊 HUENet 水下图像增强与性能评测系统")
st.caption("基于 CNN-Transformer 混合注意力机制的深度学习水下图像修复方案")

tab1, tab2, tab3 = st.tabs(["🖼️ 效果演示", "📊 批量评测统计", "📂 数据集管理"])

# --- Tab 1: 效果演示 ---
with tab1:
    st.header("单张图像实时增强")
    up_file = st.file_uploader("点击上传待处理图片", type=["jpg", "png", "jpeg"])
    
    if up_file:
        raw_pil = Image.open(up_file).convert('RGB')
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(raw_pil, caption="输入原图 (Raw)", use_container_width=True)
            
        with st.spinner("HUENet 正在推理中..."):
            tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            in_t = tf(raw_pil).unsqueeze(0).to(dev)
            
            start = time.time()
            with torch.no_grad():
                out_t = model(in_t)
            inference_time = time.time() - start
            
            out_np = out_t.detach().squeeze().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
            res_pil = Image.fromarray((out_np * 255).astype(np.uint8))
            
            with c2:
                st.image(res_pil, caption=f"增强结果 (Enhanced) - 耗时: {inference_time:.4f}s", use_container_width=True)

        # 单图指标展示
        st.subheader("📋 视觉质量分析")
        bgr = cv2.cvtColor((out_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        m1, m2 = st.columns(2)
        m1.metric("UCIQE (色彩质量)", f"{calculate_uciqe(bgr):.4f}")
        m2.metric("UIQM (感官美感)", f"{calculate_uiqm(bgr):.4f}")

# --- Tab 2: 批量评测统计 ---
with tab2:
    st.header("自动化性能评测分析")
    st.write("点击下方按钮，系统将自动遍历测试集并对比真值图，生成客观评估报表。")
    
    if st.button("🚀 执行批量测试 (UIEB Test Set)"):
        raw_path = os.path.join(dataset_root, "raw-890")
        ref_path = os.path.join(dataset_root, "reference-890")
        img_names = sorted(os.listdir(raw_path))[800:] # 测试最后90张
        
        data_logs = []
        prog = st.progress(0)
        
        for i, name in enumerate(img_names):
            # 处理
            input_pil = Image.open(os.path.join(raw_path, name)).convert('RGB')
            gt_pil = Image.open(os.path.join(ref_path, name)).convert('RGB').resize((256, 256))
            gt_np = np.array(gt_pil).astype(np.float32) / 255.0
            
            tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            out_t = model(tf(input_pil).unsqueeze(0).to(dev))
            out_np = out_t.squeeze().cpu().detach().permute(1, 2, 0).numpy().clip(0, 1)
            
            # 计算指标
            p = psnr_m(gt_np, out_np, data_range=1.0)
            s = ssim_m(gt_np, out_np, data_range=1.0, channel_axis=2)
            bgr = cv2.cvtColor((out_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            uci = calculate_uciqe(bgr)
            uiq = calculate_uiqm(bgr)
            
            data_logs.append({"文件名": name, "PSNR": p, "SSIM": s, "UCIQE": uci, "UIQM": uiq})
            prog.progress((i + 1) / len(img_names))
            
        df = pd.DataFrame(data_logs)
        
        # 指标大盘
        st.success("批量测试任务完成！")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("平均 PSNR", f"{df['PSNR'].mean():.2f}")
        k2.metric("平均 SSIM", f"{df['SSIM'].mean():.4f}")
        k3.metric("平均 UCIQE", f"{df['UCIQE'].mean():.4f}")
        k4.metric("平均 UIQM", f"{df['UIQM'].mean():.4f}")
        
        # 分布图表
        st.subheader("📊 指标统计分布")
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['PSNR'], name="PSNR (dB)"))
        fig.update_layout(title="PSNR 指标分布箱线图")
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细表格
        st.subheader("📝 详细评测报表")
        st.dataframe(df)
        st.download_button("导出评测数据(CSV)", df.to_csv(index=False), "huenet_metrics.csv", "text/csv")

# # --- Tab 3: 数据集管理 ---
# with tab3:
#     st.header("本地数据集浏览器")
#     if os.path.exists(dataset_root):
#         sub_folder = st.radio("选择浏览目录", ["raw-890 (失真图)", "reference-890 (真值图)"], horizontal=True)
#         folder_name = "raw-890" if "raw" in sub_folder else "reference-890"
#         curr_dir = os.path.join(dataset_root, folder_name)
        
#         files = sorted([f for f in os.listdir(curr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
#         if files:
#             select_idx = st.select_slider("拖动滑块查看图像", options=range(len(files)), format_func=lambda i: files[i])
#             img_path = os.path.join(curr_dir, files[select_idx])
#             st.image(Image.open(img_path), caption=f"文件名: {files[select_idx]}", width=600)
#             st.text(f"完整路径: {img_path}")
#         else:
#             st.error("该文件夹内未检测到有效图像。")
#     else:
#         st.warning("请在侧边栏配置正确的数据集根路径。")

# --- Tab 3: 数据集管理 (三图对比版) ---
# with tab3:
#     st.header("📂 本地数据集三元组对比浏览器")
#     st.markdown("同步展示：**原始退化图** | **HUENet 增强图** | **人工参考真值图**")
    
#     if os.path.exists(dataset_root):
#         # 1. 自动对齐目录
#         raw_dir = os.path.join(dataset_root, "raw-890")
#         ref_dir = os.path.join(dataset_root, "reference-890")
        
#         if not os.path.exists(raw_dir) or not os.path.exists(ref_dir):
#             st.error(f"路径不完整！请确保目录下存在 raw-890 和 reference-890 文件夹。")
#         else:
#             # 2. 获取文件列表（以raw目录为准）
#             files = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
#             if files:
#                 # 3. 交互式选择
#                 select_idx = st.select_slider(
#                     "滑动浏览数据集样本 (UIEB-890)", 
#                     options=range(len(files)), 
#                     format_func=lambda i: files[i]
#                 )
#                 filename = files[select_idx]
                
#                 # 4. 加载图像
#                 path_raw = os.path.join(raw_dir, filename)
#                 path_ref = os.path.join(ref_dir, filename)
                
#                 img_raw = Image.open(path_raw).convert('RGB')
#                 img_ref = Image.open(path_ref).convert('RGB')
                
#                 # 5. 实时推理增强图
#                 if model is not None:
#                     tf_eval = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#                     input_eval = tf_eval(img_raw).unsqueeze(0).to(dev)
                    
#                     with torch.no_grad():
#                         out_eval = model(input_eval)
                    
#                     # 后处理增强图
#                     out_eval_np = out_eval.detach().squeeze().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
#                     img_enhanced = Image.fromarray((out_eval_np * 255).astype(np.uint8))
                    
#                     # 6. 三列布局展示
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         st.image(img_raw, caption=f"原始图 (Raw): {filename}", use_container_width=True)
#                         st.info("水下原始退化特征")
                        
#                     with col2:
#                         st.image(img_enhanced, caption=f"HUENet 增强结果", use_container_width=True)
#                         # 计算当前图的指标
#                         res_bgr = cv2.cvtColor((out_eval_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
#                         u_score = calculate_uciqe(res_bgr)
#                         q_score = calculate_uiqm(res_bgr)
#                         st.success(f"UCIQE: {u_score:.3f} | UIQM: {q_score:.3f}")
                        
#                     with col3:
#                         st.image(img_ref, caption=f"真值图 (Reference)", use_container_width=True)
#                         st.info("人工优选修复基准")
                    
#                     # 7. 详细信息展示
#                     with st.expander("🔍 查看样本属性"):
#                         st.json({
#                             "文件名": filename,
#                             "原始尺寸": f"{img_raw.size[0]}x{img_raw.size[1]}",
#                             "处理分辨率": "256x256 (Model Input)",
#                             "模型设备": str(dev),
#                             "原图路径": path_raw,
#                             "参考图路径": path_ref
#                         })
#                 else:
#                     st.warning("⚠️ 模型加载失败，无法展示增强效果。")
#             else:
#                 st.error("文件夹内未发现图像文件。")
#     else:
#         st.warning("请在侧边栏输入有效的数据集根路径。")
# --- Tab 3: 数据集管理 (完美对齐版) ---
with tab3:
    st.header("📂 本地数据集三元组对比浏览器")
    st.markdown("同步展示：**原始退化图** | **HUENet 增强图** | **人工参考真值图**")
    
    if os.path.exists(dataset_root):
        raw_dir = os.path.join(dataset_root, "raw-890")
        ref_dir = os.path.join(dataset_root, "reference-890")
        
        if not os.path.exists(raw_dir) or not os.path.exists(ref_dir):
            st.error(f"路径不完整！")
        else:
            files = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            if files:
                select_idx = st.select_slider("滑动浏览数据集样本", options=range(len(files)), format_func=lambda i: files[i])
                filename = files[select_idx]
                
                # --- 1. 加载并强制统一尺寸 (核心修复) ---
                target_size = (256, 256) # 统一显示尺寸
                
                # 加载原图并调整大小
                img_raw_org = Image.open(os.path.join(raw_dir, filename)).convert('RGB')
                img_raw_disp = img_raw_org.resize(target_size, Image.LANCZOS)
                
                # 加载真值图并调整大小
                img_ref_org = Image.open(os.path.join(ref_dir, filename)).convert('RGB')
                img_ref_disp = img_ref_org.resize(target_size, Image.LANCZOS)
                
                # --- 2. 实时推理增强图 ---
                if model is not None:
                    tf_eval = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
                    input_eval = tf_eval(img_raw_org).unsqueeze(0).to(dev)
                    
                    with torch.no_grad():
                        out_eval = model(input_eval)
                    
                    out_eval_np = out_eval.detach().squeeze().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
                    img_enhanced_disp = Image.fromarray((out_eval_np * 255).astype(np.uint8))
                    
                    # --- 3. 三列布局对齐展示 ---
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(img_raw_disp, caption=f"原始图 (Raw)", use_container_width=True)
                        st.info("水下原始退化特征")
                        
                    with col2:
                        st.image(img_enhanced_disp, caption=f"HUENet 增强结果", use_container_width=True)
                        # 计算指标
                        res_bgr = cv2.cvtColor((out_eval_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        u_score = calculate_uciqe(res_bgr)
                        q_score = calculate_uiqm(res_bgr)
                        st.success(f"UCIQE: {u_score:.3f} | UIQM: {q_score:.3f}")
                        
                    with col3:
                        st.image(img_ref_disp, caption=f"真值图 (Reference)", use_container_width=True)
                        st.info("人工优选修复基准")
                    
                    # 4. 详细信息
                    with st.expander("🔍 查看样本属性"):
                        st.write(f"**文件名**: {filename}")
                        st.write(f"**原始分辨率**: {img_raw_org.size[0]} x {img_raw_org.size[1]}")
                        st.write(f"**当前显示/推理分辨率**: 256 x 256 (已对齐)")
                else:
                    st.warning("⚠️ 模型未加载")