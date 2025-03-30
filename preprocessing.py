import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocessing')

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/spectrograms', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

class Config:
    """Configuration for preprocessing"""
    # Paths
    data_dir = "/home/utopia/BMD-HS-Dataset"  # path to dataset
    train_csv = "/home/utopia/BMD-HS-Dataset/train.csv"
    metadata_csv = "/home/utopia/BMD-HS-Dataset/additional_metadata.csv"
    audio_dir = "/home/utopia/BMD-HS-Dataset/train"
    processed_dir = "data/processed"
    
    # Audio parameters
    sample_rate = 4000  # Original is 4kHz as per README
    target_duration = 20.0  # seconds (from README)
    target_sample_length = int(target_duration * sample_rate)
    
    # Spectrogram parameters
    n_fft = 1024
    win_length = 1024
    hop_length = 512
    n_mels = 128
    
    # Train/val/test split
    test_size = 0.15
    val_size = 0.15
    random_state = 42
    
    # Augmentation parameters
    time_stretch_range = (0.8, 1.2)
    pitch_shift_range = (-2, 2)  # semitones
    noise_factor_range = (0.001, 0.005)
    
    # Class columns
    class_columns = ["AS", "AR", "MR", "MS", "N"]


def load_audio_file(file_path, config):
    """
    Load and preprocess audio file using torchaudio
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != config.sample_rate:
            resampler = T.Resample(sample_rate, config.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or trim to target length
        if waveform.shape[1] < config.target_sample_length:
            # Pad with zeros
            padding = config.target_sample_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > config.target_sample_length:
            # Trim to target length
            waveform = waveform[:, :config.target_sample_length]
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform
    
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return None


def extract_features(waveform, config):
    """
    Extract mel spectrogram features from waveform
    """
    # Apply Short-time Fourier transform (STFT)
    spectrogram_transform = T.Spectrogram(
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        power=2.0,
    )
    
    # Create a mel scale transformation
    mel_transform = T.MelScale(
        n_mels=config.n_mels,
        sample_rate=config.sample_rate,
        n_stft=config.n_fft // 2 + 1,
    )
    
    # Apply STFT to get the spectrogram
    spec = spectrogram_transform(waveform)
    
    # Convert to mel scale
    mel_spec = mel_transform(spec)
    
    # Convert to dB scale (log scale)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db


def apply_augmentation(waveform, config):
    """
    Apply audio augmentations to the waveform
    """
    augmented_waveforms = [waveform]  # Start with the original
    
    # Time stretching
    stretch_factor = np.random.uniform(*config.time_stretch_range)
    time_stretch = T.TimeStretch()
    # First convert to mel spectrogram, apply stretch, then back to waveform
    spec = extract_features(waveform, config)
    stretched_spec = time_stretch(spec, stretch_factor)
    # We keep this as a feature, not back to waveform
    augmented_waveforms.append(waveform)  # Placeholder, using original as we can't easily convert back
    
    # Add background noise
    noise_factor = np.random.uniform(*config.noise_factor_range)
    noise = torch.randn_like(waveform) * noise_factor
    noisy_waveform = waveform + noise
    augmented_waveforms.append(noisy_waveform)
    
    # SpecAugment-like frequency masking on the spectrogram
    spec = extract_features(waveform, config)
    freq_mask = T.FrequencyMasking(freq_mask_param=20)
    masked_spec = freq_mask(spec)
    # Similarly, keep the spectrogram as we can't easily convert back
    augmented_waveforms.append(waveform)  # Placeholder
    
    return augmented_waveforms


def process_dataset(config):
    """
    Process the entire dataset, extract features, and save
    """
    # Load CSV files
    train_df = pd.read_csv(os.path.join(config.data_dir, config.train_csv))
    metadata_df = pd.read_csv(os.path.join(config.data_dir, config.metadata_csv))
    
    # Merge dataframes
    df = pd.merge(train_df, metadata_df, on="patient_id")
    
    # Prepare features and labels
    features = []
    labels = []
    filenames = []
    
    # Mapping for recording positions
    positions = ['sup_Mit', 'sup_Tri', 'sup_Pul', 'sup_Aor', 'sit_Mit', 'sit_Tri', 'sit_Pul', 'sit_Aor']
    recording_columns = [f"recording_{i+1}" for i in range(len(positions))]
    
    logger.info("Processing audio files and extracting features...")
    
    # Track successful files
    processed_count = 0
    failed_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):
        # Get label (multi-hot encoding)
        label = torch.tensor([row[col] for col in config.class_columns], dtype=torch.float32)
        
        # Process each recording for this patient
        for rec_col, position in zip(recording_columns, positions):
            filename = f"{row[rec_col]}.wav"
            file_path = os.path.join(config.data_dir, config.audio_dir, filename)
            
            if os.path.exists(file_path):
                waveform = load_audio_file(file_path, config)
                
                if waveform is not None:
                    # Extract features
                    mel_spec = extract_features(waveform, config)
                    
                    # Save spectrogram visualization for a few samples
                    if processed_count % 100 == 0:
                        plt.figure(figsize=(10, 4))
                        plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title(f"Mel Spectrogram - {filename}")
                        plt.tight_layout()
                        plt.savefig(os.path.join('visualizations', f"spec_{processed_count}.png"))
                        plt.close()
                    
                    # Store data
                    features.append(mel_spec.numpy())
                    labels.append(label.numpy())
                    filenames.append(filename)
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                logger.warning(f"File not found: {file_path}")
                failed_count += 1
    
    logger.info(f"Processed {processed_count} files successfully, {failed_count} files failed.")
    
    # Convert to numpy arrays using stack to ensure uniform shape
    features_array = np.stack(features)
    labels_array = np.array(labels)
    filenames_array = np.array(filenames)
    
    # Split into train, validation, and test sets
    # First split off test set
    train_val_features, test_features, train_val_labels, test_labels, train_val_filenames, test_filenames = train_test_split(
        features_array, labels_array, filenames_array, 
        test_size=config.test_size, 
        random_state=config.random_state,
        stratify=labels_array[:, -1] if np.sum(labels_array[:, -1]) > 10 else None  # Stratify by Normal class if enough samples
    )
    
    # Then split train set into train and validation
    adjusted_val_size = config.val_size / (1 - config.test_size)
    train_features, val_features, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
        train_val_features, train_val_labels, train_val_filenames,
        test_size=adjusted_val_size,
        random_state=config.random_state,
        stratify=train_val_labels[:, -1] if np.sum(train_val_labels[:, -1]) > 10 else None
    )
    
    # Save processed data
    logger.info("Saving processed data...")
    
    # Create processed data directory
    os.makedirs(config.processed_dir, exist_ok=True)
    
    # Save the data
    np.save(os.path.join(config.processed_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(config.processed_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(config.processed_dir, 'val_features.npy'), val_features)
    np.save(os.path.join(config.processed_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(config.processed_dir, 'test_features.npy'), test_features)
    np.save(os.path.join(config.processed_dir, 'test_labels.npy'), test_labels)
    
    # Save filenames for reference
    np.save(os.path.join(config.processed_dir, 'train_filenames.npy'), train_filenames)
    np.save(os.path.join(config.processed_dir, 'val_filenames.npy'), val_filenames)
    np.save(os.path.join(config.processed_dir, 'test_filenames.npy'), test_filenames)
    
    # Save config
    with open(os.path.join(config.processed_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # Print summary
    logger.info(f"Train set: {len(train_features)} samples")
    logger.info(f"Validation set: {len(val_features)} samples")
    logger.info(f"Test set: {len(test_features)} samples")
    
    # Calculate class distribution
    train_class_dist = np.sum(train_labels, axis=0)
    val_class_dist = np.sum(val_labels, axis=0)
    test_class_dist = np.sum(test_labels, axis=0)
    
    logger.info("Class distribution (train set):")
    for i, col in enumerate(config.class_columns):
        logger.info(f"{col}: {train_class_dist[i]}")
    
    # Log sample shape
    logger.info(f"Feature shape: {train_features[0].shape}")
    
    return {
        'train': (train_features, train_labels, train_filenames),
        'val': (val_features, val_labels, val_filenames),
        'test': (test_features, test_labels, test_filenames),
        'config': config
    }


class HeartSoundDataset(Dataset):
    """
    Dataset class for heart sound data
    """
    def __init__(self, features, labels, filenames=None, transform=None, augment=False, config=None):
        # Ensure features and labels are numpy arrays in case they are provided as DataFrames.
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        if isinstance(labels, pd.DataFrame):
            labels = labels.to_numpy()
            
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.filenames = filenames
        self.transform = transform
        self.augment = augment
        self.config = config
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.augment and self.config:
            # Apply SpecAugment directly on the spectrogram
            freq_mask = T.FrequencyMasking(freq_mask_param=10)
            time_mask = T.TimeMasking(time_mask_param=20)
            
            # Apply masks with 50% probability each
            if np.random.random() > 0.5:
                feature = freq_mask(feature)
            if np.random.random() > 0.5:
                feature = time_mask(feature)
        
        if self.transform:
            feature = self.transform(feature)
        
        sample = {'feature': feature, 'label': label}
        
        if self.filenames is not None:
            sample['filename'] = self.filenames[idx]
            
        return sample


def visualize_data_distribution(dataset_dict, config):
    """
    Visualize the distribution of classes in the dataset
    """
    # Extract label counts for each set
    train_labels = dataset_dict['train'][1]
    val_labels = dataset_dict['val'][1]
    test_labels = dataset_dict['test'][1]
    
    # Count instances of each class
    train_counts = np.sum(train_labels, axis=0)
    val_counts = np.sum(val_labels, axis=0)
    test_counts = np.sum(test_labels, axis=0)
    
    # Create a bar chart
    plt.figure(figsize=(12, 7))
    x = np.arange(len(config.class_columns))
    width = 0.25
    
    plt.bar(x - width, train_counts, width, label='Train')
    plt.bar(x, val_counts, width, label='Validation')
    plt.bar(x + width, test_counts, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution Across Train/Validation/Test Sets')
    plt.xticks(x, config.class_columns)
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join('visualizations', 'class_distribution.png'))
    plt.close()
    
    # Create pie charts for each set
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Train set
    axs[0].pie(train_counts, labels=config.class_columns, autopct='%1.1f%%')
    axs[0].set_title('Train Set')
    
    # Validation set
    axs[1].pie(val_counts, labels=config.class_columns, autopct='%1.1f%%')
    axs[1].set_title('Validation Set')
    
    # Test set
    axs[2].pie(test_counts, labels=config.class_columns, autopct='%1.1f%%')
    axs[2].set_title('Test Set')
    
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', 'class_distribution_pie.png'))
    plt.close()


def visualize_sample_spectrograms(dataset_dict):
    """
    Visualize sample spectrograms from the dataset
    """
    # Get sample spectrograms from each class
    train_features = dataset_dict['train'][0]
    train_labels = dataset_dict['train'][1]
    
    # Find indices for each class
    class_indices = {}
    for i in range(train_labels.shape[1]):
        class_indices[i] = np.where(train_labels[:, i] == 1)[0]
    
    # Visualize one sample from each class
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, (class_idx, indices) in enumerate(class_indices.items()):
        if len(indices) > 0:
            # Get a random sample from this class
            sample_idx = np.random.choice(indices)
            
            # Get the spectrogram
            spec = train_features[sample_idx][0]  # First channel
            
            # Plot
            im = axs[i].imshow(spec, aspect='auto', origin='lower')
            axs[i].set_title(f'Class: {Config.class_columns[class_idx]}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Mel Frequency')
            plt.colorbar(im, ax=axs[i], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', 'sample_spectrograms.png'))
    plt.close()


def main():
    """Main function"""
    logger.info("Starting preprocessing...")
    
    config = Config()
    
    # Process the dataset
    dataset_dict = process_dataset(config)
    
    # Visualize data distribution
    visualize_data_distribution(dataset_dict, config)
    
    # Visualize sample spectrograms
    visualize_sample_spectrograms(dataset_dict)
    
    logger.info("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()

