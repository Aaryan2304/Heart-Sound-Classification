import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
import librosa
import librosa.display
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dataset_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Configuration
config = {
    'data': {
        'train_path': '/home/utopia/BMD-HS-Dataset/train',
        'train_csv': '/home/utopia/BMD-HS-Dataset/train.csv',
        'output_dir': 'dataset_analysis'
    }
}

def analyze_class_distribution(train_df):
    """Analyze class distribution in train set"""
    # Get class columns (excluding ID)
    class_cols = [col for col in train_df.columns if col != 'ID']
    
    # Calculate class distribution
    train_dist = train_df[class_cols].mean()
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_cols))
    width = 0.35
    
    plt.bar(x, train_dist, width, label='Train', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Proportion')
    plt.title('Class Distribution in Train Set')
    plt.xticks(x, class_cols, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(config['data']['output_dir'], 'class_distribution.png'))
    plt.close()
    
    # Log statistics
    logging.info("\nClass Distribution:")
    for col in class_cols:
        logging.info(f"{col}:")
        logging.info(f"  Train: {train_dist[col]:.4f}")

def analyze_audio_properties(audio_dir):
    """Analyze audio properties of the dataset"""
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Initialize lists to store properties
    durations = []
    sample_rates = []
    channels = []
    
    # Analyze each audio file
    for file in tqdm(audio_files, desc='Analyzing audio files'):
        file_path = os.path.join(audio_dir, file)
        
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # Get properties
        duration = librosa.get_duration(y=y, sr=sr)
        n_channels = 1 if len(y.shape) == 1 else y.shape[0]
        
        durations.append(duration)
        sample_rates.append(sr)
        channels.append(n_channels)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Duration distribution
    plt.subplot(1, 3, 1)
    sns.histplot(durations, bins=30)
    plt.title('Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    
    # Sample rate distribution
    plt.subplot(1, 3, 2)
    sns.histplot(sample_rates, bins=30)
    plt.title('Sample Rate Distribution')
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Count')
    
    # Channel distribution
    plt.subplot(1, 3, 3)
    sns.countplot(x=channels)
    plt.title('Channel Distribution')
    plt.xlabel('Number of Channels')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['data']['output_dir'], 'audio_properties.png'))
    plt.close()
    
    # Log statistics
    logging.info("\nAudio Properties:")
    logging.info(f"Duration:")
    logging.info(f"  Mean: {np.mean(durations):.2f} seconds")
    logging.info(f"  Std: {np.std(durations):.2f} seconds")
    logging.info(f"  Min: {np.min(durations):.2f} seconds")
    logging.info(f"  Max: {np.max(durations):.2f} seconds")
    
    logging.info(f"\nSample Rate:")
    logging.info(f"  Unique values: {np.unique(sample_rates)}")
    
    logging.info(f"\nChannels:")
    logging.info(f"  Unique values: {np.unique(channels)}")

def analyze_class_correlations(train_df):
    """Analyze correlations between classes"""
    # Get class columns (excluding ID)
    class_cols = [col for col in train_df.columns if col != 'ID']
    
    # Calculate correlation matrix
    corr_matrix = train_df[class_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f'
    )
    plt.title('Class Correlation Matrix')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(config['data']['output_dir'], 'class_correlations.png'))
    plt.close()
    
    # Log high correlations
    logging.info("\nHigh Class Correlations:")
    for i in range(len(class_cols)):
        for j in range(i+1, len(class_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.3:  # Threshold for high correlation
                logging.info(f"{class_cols[i]} - {class_cols[j]}: {corr:.3f}")

def analyze_spectral_properties(audio_dir, train_df):
    """Analyze spectral properties of audio files"""
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Initialize lists to store properties
    mfccs = []
    spectral_centroids = []
    spectral_rolloffs = []
    zero_crossing_rates = []
    
    # Analyze each audio file
    for file in tqdm(audio_files, desc='Analyzing spectral properties'):
        file_path = os.path.join(audio_dir, file)
        
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Store mean values
        mfccs.append(np.mean(mfcc, axis=1))
        spectral_centroids.append(np.mean(centroid))
        spectral_rolloffs.append(np.mean(rolloff))
        zero_crossing_rates.append(np.mean(zcr))
    
    # Convert to numpy arrays
    mfccs = np.array(mfccs)
    spectral_centroids = np.array(spectral_centroids)
    spectral_rolloffs = np.array(spectral_rolloffs)
    zero_crossing_rates = np.array(zero_crossing_rates)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # MFCC distribution
    plt.subplot(2, 2, 1)
    sns.boxplot(data=pd.DataFrame(mfccs))
    plt.title('MFCC Distribution')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Value')
    
    # Spectral centroid distribution
    plt.subplot(2, 2, 2)
    sns.histplot(spectral_centroids, bins=30)
    plt.title('Spectral Centroid Distribution')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Count')
    
    # Spectral rolloff distribution
    plt.subplot(2, 2, 3)
    sns.histplot(spectral_rolloffs, bins=30)
    plt.title('Spectral Rolloff Distribution')
    plt.xlabel('Spectral Rolloff (Hz)')
    plt.ylabel('Count')
    
    # Zero crossing rate distribution
    plt.subplot(2, 2, 4)
    sns.histplot(zero_crossing_rates, bins=30)
    plt.title('Zero Crossing Rate Distribution')
    plt.xlabel('Zero Crossing Rate')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['data']['output_dir'], 'spectral_properties.png'))
    plt.close()
    
    # Log statistics
    logging.info("\nSpectral Properties:")
    logging.info(f"MFCC:")
    logging.info(f"  Mean: {np.mean(mfccs, axis=0)}")
    logging.info(f"  Std: {np.std(mfccs, axis=0)}")
    
    logging.info(f"\nSpectral Centroid:")
    logging.info(f"  Mean: {np.mean(spectral_centroids):.2f} Hz")
    logging.info(f"  Std: {np.std(spectral_centroids):.2f} Hz")
    
    logging.info(f"\nSpectral Rolloff:")
    logging.info(f"  Mean: {np.mean(spectral_rolloffs):.2f} Hz")
    logging.info(f"  Std: {np.std(spectral_rolloffs):.2f} Hz")
    
    logging.info(f"\nZero Crossing Rate:")
    logging.info(f"  Mean: {np.mean(zero_crossing_rates):.4f}")
    logging.info(f"  Std: {np.std(zero_crossing_rates):.4f}")

def main():
    # Create output directory
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    train_df = pd.read_csv(config['data']['train_csv'])
    
    # Analyze class distribution
    logging.info("Analyzing class distribution...")
    analyze_class_distribution(train_df)
    
    # Analyze audio properties
    logging.info("Analyzing audio properties...")
    analyze_audio_properties(config['data']['train_path'])
    
    # Analyze class correlations
    logging.info("Analyzing class correlations...")
    analyze_class_correlations(train_df)
    
    # Analyze spectral properties
    logging.info("Analyzing spectral properties...")
    analyze_spectral_properties(config['data']['train_path'], train_df)
    
    logging.info("Dataset analysis complete!")

if __name__ == "__main__":
    main() 
