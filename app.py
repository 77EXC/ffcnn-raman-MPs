"""
FFCNN Web Service - 拉曼微塑料在线分析平台
=============================================
Flask后端API服务

功能:
- 文件上传
- 数据验证
- 模型推理
- HFM热图生成
- 结果展示
"""

import os
import io
import uuid
import datetime
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename

from predict import (
    process_submission,
    validate_and_process_excel,
    allowed_file,
    PLASTIC_CLASSES,
    MODEL_DIR,
    UPLOAD_FOLDER
)

# ==================== Flask配置 ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件
app.config['JSON_AS_ASCII'] = False

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html',
                         title='FFCNN Raman MP Analyzer',
                         plastic_classes=PLASTIC_CLASSES)


@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html',
                         title='About - FFCNN')


@app.route('/docs')
def docs():
    """文档页面"""
    return render_template('docs.html',
                         title='Data Format - FFCNN')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    处理文件上传和推理

    请求:
    - file: Excel文件 (.xlsx)
    - email: 用户邮箱 (可选)

    返回:
    - results: 分类结果
    - heatmap: HFM热图 (base64)
    """
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Only .xlsx is allowed'}), 400

        # 保存文件
        filename = secure_filename(file.filename)
        # 添加唯一标识以避免冲突
        unique_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(file_path)

        # 处理数据
        try:
            results, heatmap_base64, config = process_submission(file_path)

            # 构建响应
            response = {
                'success': True,
                'message': f'Successfully analyzed {len(results)} spectra',
                'results': results,
                'heatmap': f'data:image/png;base64,{heatmap_base64}',
                'config': {
                    'num_classes': config.get('num_classes', len(PLASTIC_CLASSES)),
                    'plastic_classes': PLASTIC_CLASSES
                }
            }

            # 可选：保存结果
            # 这里可以添加数据库存储逻辑

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        finally:
            # 清理上传的文件
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/validate', methods=['POST'])
def validate_file():
    """
    验证文件格式 (不执行推理)

    请求:
    - file: Excel文件

    返回:
    - valid: 是否有效
    - num_spectra: 光谱数量
    - wavenumber_range: 波数范围
    - errors: 错误列表
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'valid': False,
                'error': 'Invalid file format. Only .xlsx is allowed'
            }), 400

        # 临时保存
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_validate_{filename}")
        file.save(temp_path)

        try:
            # 验证
            features, wavenumbers = validate_and_process_excel(temp_path)

            num_spectra = features.shape[0]
            wn_min = float(wavenumbers.min()) if wavenumbers is not None else None
            wn_max = float(wavenumbers.max()) if wavenumbers is not None else None

            # 检查约束
            errors = []
            if num_spectra > 1000:
                errors.append(f'Too many spectra: {num_spectra} > 1000')

            if wn_min is not None and wn_min < 500:
                errors.append(f'Wavenumber too low: {wn_min} < 500 cm⁻¹')

            if wn_max is not None and wn_max > 3500:
                errors.append(f'Wavenumber too high: {wn_max} > 3500 cm⁻¹')

            return jsonify({
                'valid': len(errors) == 0,
                'num_spectra': num_spectra,
                'wavenumber_range': [wn_min, wn_max] if wn_min is not None else None,
                'errors': errors
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500


@app.route('/sample')
def sample():
    """下载示例文件"""
    try:
        # 创建示例Excel
        import pandas as pd
        import numpy as np

        # 生成示例数据
        num_spectra = 5
        num_points = 1484  # 假设的数据点数

        data = []
        for i in range(num_spectra):
            # 横坐标 (500~3500 cm⁻¹)
            wavenumbers = np.linspace(500, 3500, num_points)
            # 模拟强度数据 (带噪声的正弦波模拟塑料特征)
            intensity = np.random.rand(num_points) * 0.3 + np.sin(wavenumbers / 500) * 0.5 + 0.5

            data.append(wavenumbers)
            data.append(intensity)

        df = pd.DataFrame(data)

        # 保存到内存
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, header=False)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='sample_raman_data.xlsx'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 错误处理 ====================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== 启动 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("FFCNN Raman Microplastics Online Analysis Platform")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Upload directory: {UPLOAD_FOLDER}")
    print("=" * 60)
    print("Starting server at http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)