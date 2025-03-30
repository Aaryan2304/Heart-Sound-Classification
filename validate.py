import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from tqdm import tqdm

from models.conformer_model import Conformer
from models.ast_model import AudioSpectrogramTransformer
from preprocessing import HeartSoundDataset
from train import config, create_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        logging.info(f"Loaded model from {checkpoint_path}")
        return True
    else:
        logging.error(f"Checkpoint {checkpoint_path} not found")
        return False

def validate_model(model, dataloader, device):
    """Validate the model and return predictions and metrics"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            # Get predictions
            preds = (probs > 0.5).float()
            
            # Store results
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return all_probs, all_preds, all_labels, metrics

def plot_confusion_matrices(all_labels, all_preds, class_names):
    """Plot confusion matrices for each class"""
    # Create directory for confusion matrices
    os.makedirs('visualizations/confusion_matrices', exist_ok=True)
    
    # Plot and save individual class confusion matrices
    for class_idx, class_name in enumerate(class_names):
        y_true = all_labels[:, class_idx]
        y_pred = all_preds[:, class_idx]
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'visualizations/confusion_matrices/{class_name}_confusion_matrix.png')
        plt.close()

def generate_classification_report(all_labels, all_preds, class_names):
    """Generate classification report for each class"""
    # Create a DataFrame to store the metrics
    report_data = []
    
    for class_idx, class_name in enumerate(class_names):
        y_true = all_labels[:, class_idx]
        y_pred = all_preds[:, class_idx]
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract metrics for the positive class (1)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        support = report['1']['support']
        
        # Add to the report data
        report_data.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    
    # Create the DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Save to CSV
    report_df.to_csv('visualizations/classification_report.csv', index=False)
    
    # Display as a styled table
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=report_df.values, 
                     colLabels=report_df.columns, 
                     cellLoc='center', 
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Classification Metrics by Class', fontsize=16)
    plt.savefig('visualizations/classification_report.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    return report_df

def main():
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Get class names
    train_df = pd.read_csv(config['data']['train_csv'])
    class_columns = [col for col in train_df.columns if col not in ['patient_id', 'ID'] + [f'recording_{i+1}' for i in range(8)]]
    
    # Load model
    model_type = config['model']['type']
    model = create_model(model_type, len(class_columns))
    
    # Define checkpoint path
    checkpoint_path = os.path.join(config['training']['model_save_dir'], f"best_model_{model_type}.pt")
    
    # Load checkpoint
    if not load_checkpoint(model, checkpoint_path):
        logging.error("Failed to load model checkpoint")
        return
    
    model = model.to(device)
    
    try:
        # Load test data
        test_features = np.load(os.path.join(config['data']['processed_dir'], 'test_features.npy'))
        test_labels = np.load(os.path.join(config['data']['processed_dir'], 'test_labels.npy'))
        test_filenames = np.load(os.path.join(config['data']['processed_dir'], 'test_filenames.npy'))
        
        logging.info(f"Test set: {len(test_features)} samples")
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Failed to load test data: {e}")
        return
    
    # Create test dataset
    test_dataset = HeartSoundDataset(
        test_features, 
        test_labels, 
        filenames=test_filenames,
        transform=None,
        augment=False,
        config=config['data']
    )
    
    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Validate model
    all_probs, all_preds, all_labels, metrics = validate_model(model, test_loader, device)
    
    # Print metrics
    logging.info("Test metrics:")
    logging.info(f"F1 Score: {metrics['f1']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(all_labels, all_preds, class_columns)
    
    # Generate classification report
    report_df = generate_classification_report(all_labels, all_preds, class_columns)
    logging.info(f"Classification report:\n{report_df}")
    
    logging.info("Validation completed successfully!")

if __name__ == "__main__":
    main() 