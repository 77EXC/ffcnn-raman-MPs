# FFCNN Raman Microplastics Online Analysis Platform

[English](README.md) | [中文](README_CN.md)

## 1. Project Overview

This is a web-based platform for analyzing Raman spectral data to identify microplastics using the FFCNN (Fourier Feature Convolutional Neural Network) framework.

### Features

- **Online Data Submission**: Upload Excel files with Raman spectral data
- **Automatic Classification**: Identifies 15+ plastic types
- **HFM Visualization**: Generates Hierarchical Feature Map heatmaps
- **Real-time Analysis**: Fast inference with pre-trained models

## 2. Data Format Specification

### File Requirements

| Parameter | Requirement |
|-----------|-------------|
| Format | `.xlsx` (Excel 2007+) |
| Layout | Every 2 columns = 1 spectrum (wavenumber, intensity) |
| Wavenumber Range | 500 ~ 3500 cm⁻¹ |
| Maximum Spectra | 1000 |
| Data Start | Cell A1 (no headers) |

### Excel Structure

```
Column 1: Spectrum 1 Wavenumbers (500, 501, 502, ..., 3500)
Column 2: Spectrum 1 Intensities (0.12, 0.45, 0.78, ...)
Column 3: Spectrum 2 Wavenumbers
Column 4: Spectrum 2 Intensities
...
```

### Supported Plastic Types

- PP (Polypropylene)
- PE (Polyethylene)
- PS (Polystyrene)
- PVC (Polyvinyl Chloride)
- PET (Polyethylene Terephthalate)
- PA (Polyamide)
- PC (Polycarbonate)
- PMMA (Polymethyl Methacrylate)
- PU (Polyurethane)
- EPS (Expanded Polystyrene)
- ABS (Acrylonitrile Butadiene Styrene)
- POM (Polyoxymethylene)
- PBT (Polybutylene Terephthalate)
- PPO (Polyphenylene Oxide)
- PPS (Polyphenylene Sulfide)

## 3. Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- 4GB+ RAM

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
web_service/
├── app.py              # Flask application
├── predict.py         # Prediction logic
├── templates/
│   ├── index.html     # Main page
│   ├── docs.html      # Documentation
│   └── about.html    # About page
├── static/
├── uploads/           # Temporary file storage
└── models/           # Link to trained models
```

## 4. Running the Server

### Local Development

```bash
cd web_service
python app.py
```

Access at: http://localhost:5000

### Production Deployment

Using Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Using Docker:

```bash
docker build -t ffcnn-web .
docker run -p 5000:5000 -v /path/to/models:/app/models ffcnn-web
```

## 5. API Usage

### Upload and Analyze

```bash
curl -X POST -F "file=@your_data.xlsx" http://localhost:5000/upload
```

Response:

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

### Validate File

```bash
curl -X POST -F "file=@your_data.xlsx" http://localhost:5000/validate
```

## 6. Output Results

### Classification Results

- **Predicted Class**: Most likely plastic type
- **Probability**: Confidence score (0-100%)
- **All Probabilities**: Full probability distribution

### HFM Visualization

Hierarchical Feature Map showing:

1. Spectral intensity patterns across all samples
2. Classification probability heatmap
3. Feature importance visualization

## 7. Configuration

### Model Path

Edit `predict.py` to specify model:

```python
MODEL_DIR = r'D:\Documents\FFCNN\models'
DEFAULT_MODEL = 'cnn_for_multi_model_res_20260605_2157.pth'
```

### Plastic Classes

Edit `predict.py`:

```python
PLASTIC_CLASSES = ['PP', 'PE', 'PS', ...]
```

## 8. Troubleshooting

### Common Issues

1. **File too large**: Maximum 16MB
2. **Invalid format**: Only .xlsx supported
3. **Out of range**: Wavenumber must be 500-3500 cm⁻¹
4. **Too many spectra**: Maximum 1000

### Logs

Check console output for error messages.

## 9. License

MIT License

## 10. Citation

```bibtex
@misc{ffcnn-raman-mps,
  author = {77EXC},
  title = {ffcnn-raman-MPs: Fourier Feature CNN for Microplastics Identification},
  year = {2024},
  url = {https://github.com/77EXC/ffcnn-raman-MPs}
}
```

## 11. Contact

- GitHub: https://github.com/77EXC/ffcnn-raman-MPs
- Issues: https://github.com/77EXC/ffcnn-raman-MPs/issues