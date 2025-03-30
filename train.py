import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
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
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Configuration
config = {
    'data': {
        'processed_dir': 'data/processed'
    },
    'training': {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'model_save_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'model': {
        'type': 'conformer',  # or 'ast'
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc='Training'):
        inputs = batch['feature'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(train_loader), f1, precision, recall

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            inputs = batch['feature'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(val_loader), f1, precision, recall

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs(config['training']['model_save_dir'], exist_ok=True)
    
    # Load preprocessed data from .npy files
    processed_dir = config['data']['processed_dir']
    train_features = np.load(os.path.join(processed_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(processed_dir, 'train_labels.npy'))
    val_features = np.load(os.path.join(processed_dir, 'val_features.npy'))
    val_labels = np.load(os.path.join(processed_dir, 'val_labels.npy'))
    test_features = np.load(os.path.join(processed_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(processed_dir, 'test_labels.npy'))
    
    logging.info(f"Train set: {train_features.shape[0]} samples")
    logging.info(f"Validation set: {val_features.shape[0]} samples")
    logging.info(f"Test set: {test_features.shape[0]} samples")
    
    # Determine number of classes from train_labels shape
    num_classes = train_labels.shape[1]
    
    # Create datasets using preprocessed data
    train_dataset = HeartSoundDataset(train_features, train_labels)
    val_dataset = HeartSoundDataset(val_features, val_labels)
    test_dataset = HeartSoundDataset(test_features, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    logging.info(f"Creating {config['model']['type']} model...")
    model = create_model(config['model']['type'], num_classes)
    model = model.to(config['training']['device'])
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    best_val_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    logging.info("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        train_loss, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, config['training']['device']
        )
        
        val_loss, val_f1, val_precision, val_recall = validate(
            model, val_loader, criterion, config['training']['device']
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_save_path = os.path.join(
                config['training']['model_save_dir'],
                f"best_model_{config['model']['type']}.pt"
            )
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Saved best model to {model_save_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['early_stopping_patience']:
            logging.info("Early stopping triggered")
            break
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_precision'], label='Train Precision')
    plt.plot(history['val_precision'], label='Val Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_recall'], label='Train Recall')
    plt.plot(history['val_recall'], label='Val Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    logging.info("Evaluating on test set...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_f1, test_precision, test_recall = validate(
        model, test_loader, criterion, config['training']['device']
    )
    
    logging.info(f"Test Results:")
    logging.info(f"Loss: {test_loss:.4f}")
    logging.info(f"F1 Score: {test_f1:.4f}")
    logging.info(f"Precision: {test_precision:.4f}")
    logging.info(f"Recall: {test_recall:.4f}")

if __name__ == "__main__":
    main()

