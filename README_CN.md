# FFCNN 拉曼微塑料在线分析平台

[English](README.md) | [中文](README_CN.md)

## 1. 项目概述

这是一个基于Web的在线平台，使用FFCNN（傅里叶特征卷积神经网络）对拉曼光谱数据进行分析，识别微塑料。

### 功能特点

- **在线数据提交**: 上传Excel格式的拉曼光谱数据
- **自动分类**: 识别15种以上塑料类型
- **HFM可视化**: 生成层次特征图热图
- **实时分析**: 预训练模型快速推理

## 2. 数据格式规范

### 文件要求

| 参数 | 要求 |
|------|------|
| 格式 | `.xlsx` (Excel 2007+) |
| 布局 | 每2列 = 1条光谱 (波数, 强度) |
| 波数范围 | 500 ~ 3500 cm⁻¹ |
| 最大光谱数 | 1000 |
| 数据起始位置 | A1单元格 (无表头) |

### Excel结构

```
第1列: 光谱1的波数 (500, 501, 502, ..., 3500)
第2列: 光谱1的强度值 (0.12, 0.45, 0.78, ...)
第3列: 光谱2的波数
第4列: 光谱2的强度值
...
```

### 支持的塑料类型

- PP (聚丙烯)
- PE (聚乙烯)
- PS (聚苯乙烯)
- PVC (聚氯乙烯)
- PET (聚对苯二甲酸乙二醇酯)
- PA (聚酰胺)
- PC (聚碳酸酯)
- PMMA (聚甲基丙烯酸甲酯)
- PU (聚氨酯)
- EPS (发泡聚苯乙烯)
- ABS (丙烯腈-丁二烯-苯乙烯)
- POM (聚甲醛)
- PBT (聚对苯二甲酸丁二醇酯)
- PPO (聚苯醚)
- PPS (聚苯硫醚)

## 3. 安装部署

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- 4GB+ 内存

### 安装依赖

```bash
pip install -r requirements.txt
```

### 目录结构

```
web_service/
├── app.py              # Flask应用
├── predict.py         # 推理逻辑
├── templates/
│   ├── index.html     # 主页面
│   ├── docs.html      # 数据格式说明
│   └── about.html    # 关于页面
├── static/
├── uploads/           # 临时文件存储
└── models/           # 链接到训练模型
```

## 4. 运行服务

### 本地开发运行

```bash
cd web_service
python app.py
```

访问地址: http://localhost:5000

### 生产环境部署

使用Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

使用Docker:

```bash
docker build -t ffcnn-web .
docker run -p 5000:5000 -v /path/to/models:/app/models ffcnn-web
```

## 5. API使用

### 上传并分析

```bash
curl -X POST -F "file=@your_data.xlsx" http://localhost:5000/upload
```

响应:

```json
{
  "success": true,
  "message": "Successfully analyzed 10 spectra",
  "results": [
    {
      "index": 0,
      "predicted": ["PP", "PE"],
      "max_class": "PP",
      "max_prob": 0.92,
      "all_probs": {"PP": 0.92, "PE": 0.05, ...}
    }
  ],
  "heatmap": "data:image/png;base64,..."
}
```

### 验证文件格式

```bash
curl -X POST -F "file=@your_data.xlsx" http://localhost:5000/validate
```

## 6. 输出结果

### 分类结果

- **预测类别**: 最可能的塑料类型
- **概率**: 置信度分数 (0-100%)
- **全概率**: 完整的概率分布

### HFM可视化

层次特征图展示:

1. 所有样本的光谱强度模式
2. 分类概率热图
3. 特征重要性可视化

## 7. 配置

### 模型路径

编辑 `predict.py`:

```python
MODEL_DIR = r'D:\Documents\FFCNN\models'
DEFAULT_MODEL = 'cnn_for_multi_model_res_20260605_2157.pth'
```

### 塑料类别

编辑 `predict.py`:

```python
PLASTIC_CLASSES = ['PP', 'PE', 'PS', ...]
```

## 8. 常见问题

### ���件问题

- 文件过大: 最大16MB
- 格式错误: 仅支持.xlsx
- 波数超范围: 必须是500-3500 cm⁻¹
- 光谱过多: 最多1000条

## 9. 许可证

MIT License

## 10. 引用

```bibtex
@misc{ffcnn-raman-mps,
  author = {77EXC},
  title = {ffcnn-raman-MPs: Fourier Feature CNN for Microplastics Identification},
  year = {2024},
  url = {https://github.com/77EXC/ffcnn-raman-MPs}
}
```

## 11. 联系方式

- GitHub: https://github.com/77EXC/ffcnn-raman-MPs
- 问题反馈: https://github.com/77EXC/ffcnn-raman-MPs/issues