"""
Inference Module for FFCNN Microplastic Identification
================================================
"""

import os
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class InferenceDataset(Dataset):
    """Dataset for inference (labels can be dummy)."""

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.zeros(10)


class FFCNNPredictor:
    """
    Predictor class for FFCNN model inference.
    """

    def __init__(self, model_path, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = config or {}

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)

        # Class names
        self.class_names = [
            'PE', 'PS', 'PET', 'PP', 'PVC',
            'PMMA', 'PC', 'PA', 'PLA', 'ABS'
        ]

    def _load_model(self, model_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        from ffcnn.model import create_model
        model = create_model(
            input_length=self.config.get('input_length', 2964),
            num_classes=self.config.get('num_classes', 10),
            hidden_channels=self.config.get('hidden_channels', 64)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @torch.no_grad()
    def predict(self, features):
        """
        Predict on input features.

        Args:
            features: np.array of shape (N, input_length)

        Returns:
            predictions: np.array of shape (N, num_classes)
            probabilities: np.array of shape (N, num_classes)
        """
        # Create dataset and loader
        dataset = InferenceDataset(features)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_probs = []

        for x, _ in loader:
            x = x.to(self.device)
            outputs = self.model(x)
            all_probs.append(outputs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        preds = (probs > 0.5).astype(int)

        return preds, probs

    def predict_excel(self, input_file, output_file, return_prob=True):
        """
        Predict from Excel file and save results.

        Args:
            input_file: Input Excel path
            output_file: Output Excel path
            return_prob: Whether to include probabilities

        Returns:
            DataFrame with predictions
        """
        # Load input data
        df = pd.read_excel(input_file)

        # Assume all columns are features (last 10 columns may be labels)
        if df.shape[1] > 2964:
            features = df.iloc[:, :-10].values
        else:
            features = df.values

        # Predict
        predictions, probabilities = self.predict(features)

        # Create results DataFrame
        results = pd.DataFrame(predictions, columns=self.class_names)

        if return_prob:
            prob_df = pd.DataFrame(probabilities, columns=[f'{c}_prob' for c in self.class_names])
            results = pd.concat([results, prob_df], axis=1)

        # Save results
        results.to_excel(output_file, index=False)

        return results


def predict(
    model_path,
    input_file,
    output_file,
    config_path='configs/default.yml',
    return_prob=True
):
    """
    Main prediction function.

    Args:
        model_path: Path to trained model (.pth)
        input_file: Path to input Excel file
        output_file: Path to save predictions
        config_path: Path to config file
        return_prob: Whether to return probabilities

    Returns:
        DataFrame with predictions
    """
    import yaml

    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create predictor
    predictor = FFCNNPredictor(model_path, config=config)

    # Predict
    results = predictor.predict_excel(input_file, output_file, return_prob)

    return results


# Example evaluation
def evaluate(model_path, test_file, config_path='configs/default.yml'):
    """
    Evaluate model on test data.
    """
    import yaml
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    from train import SpectraDataset
    from torch.utils.data import DataLoader

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load test dataset
    test_dataset = SpectraDataset(test_file)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create predictor
    predictor = FFCNNPredictor(model_path, config=config)

    # Get predictions
    all_preds = []
    all_labels = []

    for x, y in test_loader:
        preds, probs = predictor.predict(x.numpy())
        all_preds.append(preds)
        all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Metrics
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds, average='macro'):.4f}")
    print("=" * 50)

    # Per-class report
    print("\nPer-class Results:")
    print(classification_report(all_labels, all_preds, target_names=predictor.class_names))

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FFCNN Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--input', type=str, required=True, help='Input Excel file')
    parser.add_argument('--output', type=str, required=True, help='Output Excel file')
    parser.add_argument('--config', type=str, default='configs/default.yml')
    parser.add_argument('--eval', action='store_true', help='Evaluate on test set')

    args = parser.parse_args()

    if args.eval:
        evaluate(args.model, args.input, args.config)
    else:
        predict(args.model, args.input, args.output, args.config)