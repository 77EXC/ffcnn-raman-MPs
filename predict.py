"""
FFCNN Raman Microplastics Prediction Service
========================================
推理脚本：处理用户提交的拉曼光谱数据，返回分类结果和HFM热图

数据格式要求:
- Excel文件 (.xlsx)
- 每2列为一条光谱数据 (横坐标, 强度值)
- 横坐标范围: 500~3500 cm⁻¹
- 最多1000条光谱
- 数据顶格放置，无header
"""

import os
import io
import math
import base64
import pickle
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from scipy.interpolate import interp1d

# ==================== 配置 ====================
MODEL_DIR = r'D:\Documents\FFCNN\models'
UPLOAD_FOLDER = r'D:\Documents\FFCNN\web_service\uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

# 塑料类别名称 (根据你的训练数据)
PLASTIC_CLASSES = [
    'PP', 'PE', 'PS', 'PVC', 'PET',
    'PA', 'PC', 'PMMA', 'PU', 'EPS',
    'ABS', 'POM', 'PBT', 'PPO', 'PPS'
]

# 推荐的模型版本 (使用最新的)
DEFAULT_MODEL = 'cnn_for_multi_model_res_20260605_2157.pth'


class SpectraDataset(Dataset):
    """光谱数据集"""
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        return torch.tensor(feature, dtype=torch.float32)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)) + identity)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomCNN(nn.Module):
    """CNN模型 (与训练时保持一致)"""
    def __init__(self, num_points, num_classes, use_residual=False):
        super(CustomCNN, self).__init__()
        self.use_residual = use_residual
        self.features = nn.ModuleList()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.features.append(nn.Sequential(self.conv1, self.bn1, self.relu, self.pool1))

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.features.append(nn.Sequential(self.conv2, self.bn2, self.relu, self.pool2))

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        for _ in range(3):
            self.features.append(ResnetBlock_remove_IN1D(dim=64, dilation=1))

        if self.use_residual:
            self.residual_block = ResidualBlock(in_channels=64, out_channels=64)
            self.features.append(self.residual_block)

        self.tanh = nn.Tanh()
        final_dim = int(math.ceil(num_points / 2))
        self.fc1 = nn.Linear(64 * final_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.tanh(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)


class ResnetBlock_remove_IN1D(nn.Module):
    """1D残差块 (与ffc_use.py保持一致)"""
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_remove_IN1D, self).__init__()
        from ffc import FFC_BN_ACT1D

        self.ffc1 = FFC_BN_ACT1D(dim, dim, 3, 0.75, 0.75, stride=1, padding=1,
                                 dilation=dilation, groups=1, bias=False,
                                 norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU,
                                 enable_lfu=False)

        self.ffc2 = FFC_BN_ACT1D(dim, dim, 3, 0.75, 0.75, stride=1, padding=1,
                                 dilation=1, groups=1, bias=False,
                                 norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU,
                                 enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output
        return output


# ==================== 工具函数 ====================
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path=None):
    """加载模型和配置"""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL)

    # 加载配置
    cfg_path = model_path.replace('.pth', '_cfg.pkl')
    if not os.path.exists(cfg_path):
        # 尝试查找配置
        model_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path).replace('.pth', '')
        for f in os.listdir(model_dir):
            if f.startswith(base_name) and f.endswith('_cfg.pkl'):
                cfg_path = os.path.join(model_dir, f)
                break

    with open(cfg_path, 'rb') as f:
        config = pickle.load(f)

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(config['num_points'], config['num_classes'], use_residual=config['use_residual'])
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'] if 'state_dict' in torch.load(model_path, map_location=device)
                     else torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config, device


def validate_and_process_excel(filepath):
    """
    验证并处理Excel文件

    返回:
    - normalized_features: 归一化后的特征 (N, num_points)
    - wavenumbers: 波数数组
    """
    df = pd.read_excel(filepath, header=None)
    print(f"Excel数据形状: {df.shape}")

    # 验证格式：应该是每2列一条光谱
    num_cols = df.shape[1]
    if num_cols % 2 != 0:
        raise ValueError(f"列数必须是偶数，当前为 {num_cols}")

    num_spectra = num_cols // 2

    # 提取数据
    # 假设奇数列为横坐标(波数)，偶数列为纵坐标(强度)
    spectra_data = []
    wavenumbers = None

    for i in range(num_spectra):
        wn_col = i * 2       # 横坐标列
        int_col = i * 2 + 1   # 纵坐标列

        wavenumber = df.iloc[:, wn_col].values
        intensity = df.iloc[:, int_col].values

        # 验证波数范围
        if wavenumbers is None:
            wavenumbers = wavenumber
            valid_mask = (wavenumber >= 500) & (wavenumber <= 3500)
        else:
            valid_mask = valid_mask & (wavenumber >= 500) & (wavenumber <= 3500)

        spectra_data.append(intensity)

    # 检查波数范围
    if not valid_mask.all():
        print("警告: 部分数据波数不在500~3500 cm⁻¹范围内")

    # 转换为numpy数组
    features = np.array(spectra_data)
    print(f"原始特征形状: {features.shape}")

    # 归一化
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    # 避免除零
    features_std[features_std == 0] = 1
    normalized_features = (features - features_mean) / features_std

    # 调整维度以匹配模型输入
    # 需要与训练时相同的点数
    # 注意：这里需要根据实际模型配置调整

    return normalized_features, wavenumbers


def predict(model, features, device, config):
    """执行预测"""
    # 创建数据加载器
    dataset = SpectraDataset(features)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.unsqueeze(1).to(device)
            outputs = model(batch)
            all_outputs.append(outputs.cpu())

    all_outputs = torch.cat(all_outputs)

    # 解析结果
    results = []
    threshold = 0.5

    for i in range(len(all_outputs)):
        probs = all_outputs[i].numpy()
        predictions = (probs > threshold).astype(int)

        # 获取预测的类别
        predicted_classes = []
        for j, pred in enumerate(predictions):
            if pred == 1:
                predicted_classes.append(PLASTIC_CLASSES[j])

        # 获取最高概率的类别
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]

        results.append({
            'index': i,
            'predicted': predicted_classes if predicted_classes else ['Unknown'],
            'max_class': PLASTIC_CLASSES[max_idx],
            'max_prob': float(max_prob),
            'all_probs': {PLASTIC_CLASSES[j]: float(probs[j]) for j in range(len(probs))}
        })

    return results


def generate_hfm_heatmap(features, wavenumbers, results, output_path=None):
    """
    生成HFM (Hierarchical Feature Map) 可视化热图

    参数:
    - features: 归一化后的特征 (N, num_points)
    - wavenumbers: 波数数组
    - results: 预测结果
    - output_path: 输出路径

    返回:
    - base64编码的图像
    """
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上图：光谱热图
    ax1 = axes[0]
    num_spectra = min(len(features), 20)  # 最多显示20条

    # 创建热图数据
    heatmap_data = features[:num_spectra]

    sns.heatmap(heatmap_data, ax=ax1, cmap='viridis',
                xticklabels=min(50, len(wavenumbers)),
                yticklabels=min(num_spectra, 10))
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Sample Index')
    ax1.set_title('Raman Spectra Heatmap (HFM)')

    # 下图：分类概率热图
    ax2 = axes[1]

    # 构建概率矩阵
    probs_matrix = []
    for r in results[:num_spectra]:
        row = [r['all_probs'].get(c, 0) for c in PLASTIC_CLASSES]
        probs_matrix.append(row)

    probs_matrix = np.array(probs_matrix)

    sns.heatmap(probs_matrix, ax=ax2, annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=PLASTIC_CLASSES,
                yticklabels=[f'Sample {i}' for i in range(num_spectra)])
    ax2.set_xlabel('Plastic Type')
    ax2.set_ylabel('Sample')
    ax2.set_title('Classification Probability Heatmap')

    plt.tight_layout()

    # 保存到内存
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(img_buffer.getvalue())

    return img_base64


def process_submission(file_path, model_path=None):
    """
    处理用户提交的数据

    参数:
    - file_path: Excel文件路径
    - model_path: 模型路径 (可选)

    返回:
    - results: 预测结果
    - heatmap_base64: HFM热图base64编码
    """
    # 加载模型
    model, config, device = load_model(model_path)

    # 处理数据
    features, wavenumbers = validate_and_process_excel(file_path)

    # 调整特征维度以匹配模型
    target_points = config['num_points']
    current_points = features.shape[1]

    if current_points != target_points:
        # 需要插值或截断
        from scipy.interpolate import interp1d

        # 创建原始波数索引
        original_indices = np.linspace(0, 1, current_points)
        target_indices = np.linspace(0, 1, target_points)

        interpolated_features = []
        for i in range(len(features)):
            f = interp1d(original_indices, features[i], kind='linear')
            interpolated_features.append(f(target_indices))

        features = np.array(interpolated_features)
        print(f"调整后的特征形状: {features.shape}")

    # 预测
    results = predict(model, features, device, config)

    # 生成HFM热图
    heatmap_base64 = generate_hfm_heatmap(features, wavenumbers, results)

    return results, heatmap_base64, config


# ==================== 主程序 ====================
if __name__ == '__main__':
    import math

    # 测试
    test_file = r'D:\Documents\FFCNN\input_data\test_data.xlsx'

    if os.path.exists(test_file):
        print("开始处理测试数据...")
        results, heatmap, config = process_submission(test_file)

        print(f"\n预测结果 (前5条):")
        for r in results[:5]:
            print(f"  样本 {r['index']}: {r['max_class']} ({r['max_prob']:.2%})")

        print(f"\nHFM热图已生成")
    else:
        print(f"测试文件不存在: {test_file}")