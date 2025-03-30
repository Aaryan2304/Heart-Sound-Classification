# Heart Sound Classification using Deep Learning

This project implements deep learning models for classifying heart sounds using the BMD-HS-Dataset. The project includes two state-of-the-art models: Conformer and Audio Spectrogram Transformer (AST), along with comprehensive data analysis and visualization tools.

## Project Structure

```
.
├── models/
│   ├── conformer_model.py    # Conformer model implementation
│   └── ast_model.py         # Audio Spectrogram Transformer implementation
├── preprocessing.py         # Data preprocessing and augmentation
├── train.py                # Training script
├── evaluate.py             # Model evaluation script
├── visualize.py            # Visualization tools
├── analyze_dataset.py      # Dataset analysis tools
└── requirements.txt        # Project dependencies
```

## Features

- Multi-label classification of heart sounds
- Two state-of-the-art models:
  - Conformer: Combines self-attention and convolution for sequence modeling
  - Audio Spectrogram Transformer: Vision transformer adapted for audio spectrograms
- Comprehensive data analysis:
  - Class distribution analysis
  - Audio properties analysis
  - Spectral feature analysis
  - Class correlation analysis
- Visualization tools:
  - Training history plots
  - Attention maps
  - ROC curves
  - Confusion matrices
- Model evaluation metrics:
  - F1 score
  - Precision
  - Recall
  - ROC-AUC

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-sound-classification.git
cd heart-sound-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Analysis

To analyze the dataset and generate insights:
```bash
python analyze_dataset.py
```

This will create visualizations and statistics in the `dataset_analysis` directory.

### Training

To train a model:
```bash
python train.py
```

The script will:
- Load and preprocess the data
- Train the model (Conformer or AST)
- Save the best model checkpoint
- Generate training history plots

### Evaluation

To evaluate trained models:
```bash
python evaluate.py
```

This will:
- Load the trained models
- Evaluate on the test set
- Generate performance metrics and visualizations
- Compare model performance

### Visualization

To visualize model predictions and attention maps:
```bash
python visualize.py
```

This will create visualizations in the `visualizations` directory.

## Model Architecture

### Conformer

The Conformer model combines self-attention and convolution for sequence modeling:
- Multi-head self-attention
- Convolution module
- Feed-forward networks
- Layer normalization and residual connections

### Audio Spectrogram Transformer (AST)

The AST model adapts the Vision Transformer for audio spectrograms:
- Patch embedding
- Positional encoding
- Multi-head self-attention
- MLP blocks
- Global average pooling

## Dataset

The BMD-HS-Dataset contains heart sound recordings with the following classes:
- Normal
- Aortic Stenosis
- Mitral Regurgitation
- Mitral Stenosis
- Mitral Valve Prolapse

## Results

The models achieve the following performance metrics:

| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|---------|
| Conformer | X.XX | X.XX | X.XX |
| AST | X.XX | X.XX | X.XX |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BMD-HS-Dataset creators and contributors
- PyTorch team for the deep learning framework
- Librosa team for audio processing tools 