import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from conformer_model import Conformer
from ast_model import AudioSpectrogramTransformer
from preprocessing import HeartSoundDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Configuration
config = {
    'data': {
        'train_path': '/home/utopia/BMD-HS-Dataset/train',
        'train_csv': '/home/utopia/BMD-HS-Dataset/train.csv',
        'processed_dir': 'data/processed',
        'sample_rate': 22050,
        'duration': 5,  # seconds
        'n_mels': 128,
        'hop_length': 512,
        'n_fft': 2048
    },
    'models': {
        'conformer': {
            'checkpoint': 'checkpoints/best_model_conformer.pt',
            'config': {
                'input_dim': 128,
                'num_heads': 8,
                # Removed 'ffn_dim' since Conformer doesn't expect it.
                'num_layers': 6,
                'dropout': 0.1
            }
        },
        'ast': {
            'checkpoint': 'checkpoints/best_model_ast.pt',
            'config': {
                'img_size': (128, 1024),
                'patch_size': (16, 16),
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0,
                'drop_rate': 0.1,
                'attn_drop_rate': 0.1
            }
        }
    },
    'evaluation': {
        'batch_size': 32,
        'output_dir': 'evaluation_results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
}

def create_model(model_name, num_classes):
    """Create model based on configuration"""
    if model_name == 'conformer':
        model = Conformer(
            input_dim=config['models']['conformer']['config']['input_dim'],
            num_heads=config['models']['conformer']['config']['num_heads'],
            num_layers=config['models']['conformer']['config']['num_layers'],
            dropout=config['models']['conformer']['config']['dropout'],
            num_classes=num_classes
        )
    else:  # ast
        model = AudioSpectrogramTransformer(
            img_size=config['models']['ast']['config']['img_size'],
            patch_size=config['models']['ast']['config']['patch_size'],
            embed_dim=config['models']['ast']['config']['embed_dim'],
            depth=config['models']['ast']['config']['depth'],
            num_heads=config['models']['ast']['config']['num_heads'],
            mlp_ratio=config['models']['ast']['config']['mlp_ratio'],
            drop_rate=config['models']['ast']['config']['drop_rate'],
            attn_drop_rate=config['models']['ast']['config']['attn_drop_rate'],
            num_classes=num_classes
        )
    return model

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            inputs = batch['feature'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(test_loader),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': (confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1)) 
                             if all_labels.shape[1] > 1 
                             else confusion_matrix(all_labels, all_preds))
    }
    
    return metrics, all_probs

def plot_roc_curves(all_probs, all_labels, classes, save_path):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_models(model_results):
    """Compare performance of different models"""
    metrics = ['f1', 'precision', 'recall']
    models = list(model_results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [model_results[model][metric] for model in models]
        ax.bar(models, values)
        ax.set_title(f'{metric.capitalize()} Score')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['evaluation']['output_dir'], 'model_comparison.png'))
    plt.close()

def main():
    os.makedirs(config['evaluation']['output_dir'], exist_ok=True)
    
    logging.info("Loading preprocessed test data...")
    test_features = np.load(os.path.join(config['data']['processed_dir'], 'test_features.npy'))
    test_labels = np.load(os.path.join(config['data']['processed_dir'], 'test_labels.npy'))
    test_filenames = np.load(os.path.join(config['data']['processed_dir'], 'test_filenames.npy'))
    
    logging.info(f"Test set: {len(test_features)} samples")
    
    train_df = pd.read_csv(config['data']['train_csv'])
    class_columns = [col for col in train_df.columns if col not in ['patient_id', 'ID'] + [f'recording_{i+1}' for i in range(8)]]
    
    test_dataset = HeartSoundDataset(
        features=test_features,
        labels=test_labels,
        filenames=test_filenames
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    num_classes = len(class_columns)
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Classes: {class_columns}")
    
    criterion = nn.BCEWithLogitsLoss()
    
    model_results = {}
    for model_name, model_config in config['models'].items():
        if not os.path.exists(model_config['checkpoint']):
            logging.warning(f"Model checkpoint {model_config['checkpoint']} not found. Skipping evaluation.")
            continue
            
        logging.info(f"Evaluating {model_name} model...")
        
        model = create_model(model_name, num_classes)
        model.load_state_dict(torch.load(model_config['checkpoint']))
        model = model.to(config['evaluation']['device'])
        
        metrics, all_probs = evaluate_model(
            model, test_loader, criterion, config['evaluation']['device']
        )
        
        model_results[model_name] = metrics
        
        plot_roc_curves(
            all_probs,
            test_labels,
            class_columns,
            os.path.join(config['evaluation']['output_dir'], f'{model_name}_roc_curves.png')
        )
        
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_columns,
            os.path.join(config['evaluation']['output_dir'], f'{model_name}_confusion_matrix.png')
        )
        
        logging.info(f"\n{model_name} Results:")
        logging.info(f"Loss: {metrics['loss']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
    
    if len(model_results) > 1:
        compare_models(model_results)
    
    logging.info("Evaluation complete!")

if __name__ == "__main__":
    main()

