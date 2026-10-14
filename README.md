# UENet 水下图像增强系统

基于 CNN-Transformer 混合注意力机制的深度学习水下图像修复方案，针对水下图像色偏、雾化、低对比度、细节模糊等退化问题，提供完整的训练、测试、多模型对比与可视化演示全流程工具链。

## ✨ 核心特性

- **混合架构设计**：CNN 残差网络提取局部细节特征，Transformer 瓶颈层建模全局语义信息，兼顾增强精度与语义一致性
- **多数据集训练**：支持 UIEB、EUVP 数据集联合训练，自动划分训练 / 验证集
- **专业评估体系**：覆盖全参考指标（PSNR / SSIM）与水下专用无参考指标（UCIQE / UIQM）
- **多算法对比**：内置 CLAHE、FUnIEGAN、WaterNet 对比基准，一键生成横向对比报表
- **断点续训支持**：自动保存训练状态，中断后可无缝恢复训练进度
- **可视化交互系统**：基于 Streamlit 的 Web 演示平台，支持单图实时推理、批量评测、数据集三元组对比浏览

## 📦 环境依赖

- Python 3.12+
- PyTorch 2.2.0+ / CUDA 推荐
- 完整依赖见 `requirements.txt`

一键安装依赖：

```
pip install -r requirements.txt
```

## 📂 项目目录结构

```
huenet/
├── checkpoints/          # 模型权重存放目录
├── data/                 # 数据集目录（已加入 .gitignore，不上传）
│   └── UIEB/
│       ├── raw-890/      # 退化水下图像集
│       └── reference-890/# 真值参考图像集
├── datasets/             # 数据集加载模块
│   └── dataset.py
├── models/               # 模型架构定义
│   └── huenet.py
├── results/              # 单模型测试输出结果（已加入 .gitignore）
├── comparison_results/   # 多模型对比实验输出（已加入 .gitignore）
├── utils/                # 工具函数目录
├── venv/                 # Python 虚拟环境（已加入 .gitignore）
├── train.py              # 模型训练脚本
├── test.py               # 单模型测试与指标计算脚本
├── run_comparison.py     # 多算法横向对比实验脚本
├── app.py                # Streamlit 可视化演示系统
├── requirements.txt      # 依赖包清单
├── .gitignore            # Git 忽略规则
└── .gitattributes        # 换行符统一配置
```

## 🚀 快速开始

### 1. 数据集准备

- 下载 UIEB 数据集，放入 `./data/UIEB/` 目录，确保包含 `raw-890` 和 `reference-890` 两个子文件夹
- （可选）下载 EUVP 数据集用于联合训练，放入 `./data/EUVP/` 目录

### 2. 模型训练

```
python train.py
```

- 默认使用 UIEB 前 800 张作为训练集，后 90 张作为验证集
- 自动断点续训，加载 `checkpoints/latest_model.pth` 恢复训练状态
- 验证集最优模型自动保存为 `checkpoints/best_model.pth`

### 3. 模型测试

```
python test.py
```

- 在 UIEB 测试集上计算 PSNR、SSIM、UCIQE、UIQM 四项指标
- 增强后的图像自动保存至 `./results/` 目录

### 4. 多模型对比实验

```
python run_comparison.py
```

- 同步测试 CLAHE、FUnIEGAN、WaterNet、HUENet 四种增强方法
- 控制台输出指标对比表格，结果图像与 CSV 报表保存至 `./comparison_results/`

### 5. 启动可视化演示系统

```
streamlit run app.py
```

启动后浏览器自动打开 Web 界面，支持三大功能：

- 单张图像上传、实时增强与质量指标分析
- 批量自动化测试与指标分布统计
- 数据集「原图 - 增强图 - 真值图」三元组对比浏览

## 📊 评估指标说明

表格

| 指标 | 类型 | 说明 | 优劣趋势 |
| --- | --- | --- | --- |
| PSNR | 全参考 | 峰值信噪比，衡量图像保真度与噪声水平 | 越高越好 |
| SSIM | 全参考 | 结构相似性，衡量亮度、对比度、结构的一致性 | 越接近 1 越好 |
| UCIQE | 无参考 | 水下色彩质量指标，评价色度、饱和度与对比度均匀性 | 越高越好 |
| UIQM | 无参考 | 水下图像综合质量，融合色彩、锐度、对比度三个维度 | 越高越好 |

## ❓ 常见问题

1. **路径报错**：项目默认使用相对路径，确保数据集与权重放置在对应目录，或修改脚本中路径配置
2. **权重加载失败**：脚本已自动处理多卡训练权重的 `module.` 前缀，若仍报错可检查权重文件键名结构
3. **Git 换行符警告**：项目已配置 `.gitattributes` 统一 LF 格式，正常提交即可
4. **GitHub 推送连接失败**：建议配置代理或切换 SSH 方式推送，详见 Git 官方文档

## 📄 说明

本项目仅供学术研究与学习使用。
