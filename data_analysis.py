import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Path to the dataset
dataset_path = '.'
train_csv_path = os.path.join(dataset_path, 'train.csv')
metadata_csv_path = os.path.join(dataset_path, 'additional_metadata.csv')
audio_folder = os.path.join(dataset_path, 'train')

# Read CSV files
train_df = pd.read_csv(train_csv_path)
metadata_df = pd.read_csv(metadata_csv_path)

# Print dataset information
print(f"Number of patients: {len(train_df)}")
print(f"Classes: AS, AR, MR, MS, N (Normal)")

# Analyze class distribution
class_columns = ['AS', 'AR', 'MR', 'MS', 'N']
for col in class_columns:
    print(f"Number of patients with {col}: {train_df[col].sum()}")

# Analyze one audio file to get its properties
sample_audio_path = os.path.join(audio_folder, os.listdir(audio_folder)[0])
print(f"\nAnalyzing sample audio file: {sample_audio_path}")
try:
    sample_rate, audio_data = wavfile.read(sample_audio_path)
    duration = len(audio_data) / sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Channels: {1 if len(audio_data.shape) == 1 else audio_data.shape[1]}")
    print(f"Shape: {audio_data.shape}")
    print(f"Data type: {audio_data.dtype}")
    print(f"Min value: {audio_data.min()}")
    print(f"Max value: {audio_data.max()}")
    print(f"Mean value: {audio_data.mean()}")
except Exception as e:
    print(f"Error reading audio file: {e}")

# Check for potential issues
print("\nPotential issues:")
if len(os.listdir(audio_folder)) != sum([len(train_df[col]) for col in train_df.columns if 'recording' in col]):
    print("- Mismatch between number of audio files and recordings listed in train.csv")

# Are there missing files?
all_recording_files = []
for col in [c for c in train_df.columns if 'recording' in c]:
    all_recording_files.extend(train_df[col].tolist())
    
missing_files = [file for file in all_recording_files if not os.path.exists(os.path.join(audio_folder, f"{file}.wav"))]
if missing_files:
    print(f"- {len(missing_files)} missing audio files")
    print(f"  First few missing: {missing_files[:5]}")

print("\nDone with analysis") 