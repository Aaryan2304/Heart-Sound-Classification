import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

from conformer_model import Conformer
from ast_model import AudioSpectrogramTransformer
from preprocessing import HeartSoundDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    'model': {
        'type': 'conformer',  # or 'ast'
        'checkpoint': 'checkpoints/best_model_conformer.pt',  # or 'ast'
        'conformer': {
            'input_dim': 128,
            'num_heads': 8,
            # Removed 'ffn_dim' since Conformer doesn't expect it.
            'num_layers': 6,
            'dropout': 0.1
        },
        'ast': {
            'img_size': (128, 1024),
            'patch_size': (16, 16),
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'drop_rate': 0.1,
            'attn_drop_rate': 0.1
        }
    },
    'visualization': {
        'num_samples': 10,
        'output_dir': 'visualizations',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
}

def create_model(model_type, num_classes):
    """Create model based on configuration"""
    if model_type == 'conformer':
        model = Conformer(
            input_dim=config['model']['conformer']['input_dim'],
            num_heads=config['model']['conformer']['num_heads'],
            num_layers=config['model']['conformer']['num_layers'],
            dropout=config['model']['conformer']['dropout'],
            num_classes=num_classes
        )
    else:  # ast
        model = AudioSpectrogramTransformer(
            img_size=config['model']['ast']['img_size'],
            patch_size=config['model']['ast']['patch_size'],
            embed_dim=config['model']['ast']['embed_dim'],
            depth=config['model']['ast']['depth'],
            num_heads=config['model']['ast']['num_heads'],
            mlp_ratio=config['model']['ast']['mlp_ratio'],
            drop_rate=config['model']['ast']['drop_rate'],
            attn_drop_rate=config['model']['ast']['attn_drop_rate'],
            num_classes=num_classes
        )
    return model

def plot_attention_maps(model, input_tensor, attention_maps, save_path):
    """Plot attention maps for each layer"""
    num_layers = len(attention_maps)
    
    # Create a figure with subplots for each layer
    fig, axes = plt.subplots(num_layers, 1, figsize=(15, 5*num_layers))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    for layer_idx, (ax, attn) in enumerate(zip(axes, attention_maps)):
        # Average attention across heads
        avg_attn = attn.mean(dim=1)  # (B, N, N)
        sns.heatmap(
            avg_attn[0].cpu().numpy(),
            ax=ax,
            cmap='viridis',
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(f'Layer {layer_idx + 1} Attention Map')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram_and_attention(spectrogram, attention_maps, save_path, title=None):
    """Plot spectrogram with attention overlay"""
    num_layers = len(attention_maps)
    
    # Create figure with subplots; ensure axes is always a list
    fig, axes = plt.subplots(num_layers + 1, 1, figsize=(15, 5*(num_layers + 1)))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    # Squeeze the spectrogram to get a 2D array.
    spec = spectrogram.squeeze(0)
    if spec.ndim == 3:
        spec = spec.squeeze(0)
    
    # Plot spectrogram
    sns.heatmap(
        spec.cpu().numpy(),
        ax=axes[0],
        cmap='viridis',
        xticklabels=False,
        yticklabels=False
    )
    axes[0].set_title('Mel Spectrogram')
    
    # Plot attention maps for each layer
    for layer_idx, (ax, attn) in enumerate(zip(axes[1:], attention_maps)):
        avg_attn = attn.mean(dim=1)  # (B, N, N)
        sns.heatmap(
            avg_attn[0].cpu().numpy(),
            ax=ax,
            cmap='viridis',
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(f'Layer {layer_idx + 1} Attention Map')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, dataset, num_samples, output_dir):
    """Visualize model predictions and attention maps"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Visualizing samples'):
            batch = dataset[idx]
            if isinstance(batch, dict):
                feature = batch['feature']
                label = batch['label']
            else:
                feature, label = batch
            
            input_tensor = feature.unsqueeze(0).to(config['visualization']['device'])
            
            # Get predictions and attention maps
            if config['model']['type'] == 'ast':
                outputs, attention_maps = model(input_tensor, return_attention=True)
            else:
                outputs = model(input_tensor)
                attention_maps = model.get_attention_maps(input_tensor) if hasattr(model, 'get_attention_maps') else []
            
            preds = (torch.sigmoid(outputs) > 0.5).int()
            true_labels = [i for i, l in enumerate(label) if l == 1]
            pred_labels = [i for i, p in enumerate(preds[0]) if p == 1]
            title = f'True: {true_labels}, Pred: {pred_labels}'
            
            save_path = os.path.join(output_dir, f'sample_{idx}.png')
            plot_spectrogram_and_attention(input_tensor, attention_maps, save_path, title)
            
            if attention_maps:
                save_path = os.path.join(output_dir, f'attention_maps_{idx}.png')
                plot_attention_maps(model, input_tensor, attention_maps, save_path)

def main():
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
    
    num_classes = len(class_columns)
    logging.info(f"Number of classes: {num_classes}")
    
    logging.info(f"Creating {config['model']['type']} model...")
    model = create_model(config['model']['type'], num_classes)
    
    if not os.path.exists(config['model']['checkpoint']):
        logging.error(f"Model checkpoint {config['model']['checkpoint']} not found. Exiting.")
        return
    
    logging.info(f"Loading checkpoint from {config['model']['checkpoint']}")
    model.load_state_dict(torch.load(config['model']['checkpoint']))
    model = model.to(config['visualization']['device'])
    
    logging.info("Visualizing predictions and attention maps...")
    visualize_predictions(
        model,
        test_dataset,
        config['visualization']['num_samples'],
        config['visualization']['output_dir']
    )
    
    logging.info("Visualization complete!")

if __name__ == "__main__":
    main()

