# Heart Sound Classification Project

This project implements deep learning models for heart sound classification using the BMD-HS-Dataset, focusing on classifying different heart valve conditions (AS, AR, MR, MS) and normal heart sounds.

## Project Structure

```
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── data_analysis.py           # Dataset analysis and visualization
├── run_preprocessing.py       # Data preprocessing pipeline
├── dataset.py                 # PyTorch dataset implementation
├── conformer_model.py         # Conformer model implementation
├── ast_model.py               # Audio Spectrogram Transformer model
├── run_training.py            # Training pipeline
└── run_pipeline.py            # End-to-end pipeline
```

## Setup

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the BMD-HS-Dataset and place it in your project directory:
   - The dataset should be in a subdirectory called `BMD-HS-Dataset-main` with:
     - `train/` directory containing audio files
     - `train.csv` with labels
     - `additional_metadata.csv` with metadata

## Usage

### 1. Analyze the Dataset

Run data analysis to understand the dataset characteristics and validate its integrity:

```bash
python data_analysis.py
```

This will create visualizations and analysis reports in the `dataset_analysis` directory.

### 2. Preprocess the Data

Preprocess the audio files, extract features, and split into train/validation/test sets:

```bash
python run_preprocessing.py
```

Processed data will be saved to the `data/processed` directory.

### 3. Train Models

Train a model (Conformer by default):

```bash
python run_training.py
```

To specify the model type (Conformer or AST):

```bash
python run_training.py --model-type=ast
```

The trained models will be saved to the `checkpoints` directory.

### 4. Run the Complete Pipeline

To run the complete preprocessing, training, and evaluation pipeline:

```bash
python run_pipeline.py
```

This will run all steps of the pipeline sequentially. You can also run specific parts:

```bash
python run_pipeline.py --preprocess --train --model-type=conformer
```

## Models

The project implements two different deep learning architectures:

1. **Conformer**: Combines self-attention and convolution for audio classification
2. **Audio Spectrogram Transformer (AST)**: Vision transformer adapted for audio spectrograms

## Customization

You can modify the configuration in `config.py` to change:

- Audio parameters (sample rate, duration, etc.)
- Model hyperparameters
- Training parameters (batch size, learning rate, etc.)
- Data augmentation settings

## Outputs

- Preprocessed data: `data/processed/`
- Visualizations: `visualizations/`
- Dataset analysis: `dataset_analysis/`
- Model checkpoints: `checkpoints/`
- Evaluation results: `evaluation_results/`

## Troubleshooting

If you encounter issues:

1. Check the log outputs for detailed error messages
2. Ensure the dataset structure matches the expected format
3. Verify that all dependencies are installed correctly
4. Make sure you have sufficient disk space for processed data

## License

This project is licensed under the MIT License - see LICENSE for details. 