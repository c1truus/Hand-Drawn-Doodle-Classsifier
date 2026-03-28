# Hand-Drawn-Doodle-Classifier

A PyTorch-based neural network classifier for MNIST handwritten digit recognition.

## Project Structure

```
Hand-Drawn-Doodle-Classsifier/
├── src/               # Source code and notebooks
├── data/              # MNIST dataset
├── images/            # Documentation images
├── doc/               # References and documentation
└── misc/              # Utility scripts
```

## Environment

Tested on Ubuntu 24.04.4 LTS with:
- Python 3.12.3
- PyTorch 2.7.1 (CUDA 11.8)
- NVIDIA GTX 1060 6GB
- Driver 580

## Setup

### Virtual Environment

```bash
python -m venv .env
source .env/bin/activate
```

### PyTorch Installation

For GTX 1060 (compute capability 6.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Dataset Preparation

1. Download MNIST from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
2. Extract files to `data/`
3. Run preprocessing:

```bash
cd data
python split.py      # Create train/valid/test splits
python load-mnist.py # Verify data loading
python sample.py     # Visualize samples
```

At the end you must have this exact file structure :

```bash
$ ll
total 40
drwxrwxr-x  5 4096 Mar 28 15:56 ./
drwxrwxr-x 10 4096 Mar 28 19:28 ../
-rw-rw-r--  1 2594 Mar 28 15:48 load-mnist.py
-rw-rw-r--  1 5289 Mar 28 15:56 sample.py
-rw-rw-r--  1 5558 Mar 28 15:51 split.py
drwxrwxr-x  5 4096 Mar 28 15:51 test/
drwxrwxr-x  5 4096 Mar 28 15:51 train/
drwxrwxr-x  5 4096 Mar 28 15:51 valid/
```

## Model Architecture

- Input: 784 (28×28 pixels)
- Hidden Layer 1: 1024 neurons (LeakyReLU)
- Hidden Layer 2: 1024 neurons (LeakyReLU)
- Output: 10 neurons (digit classes)

Total parameters: ~2.1M

## Results

| Dataset | Accuracy |
|---------|----------|
| Training | 99.8% |
| Validation | 97.8% |
| Test | 98.0% |

## Usage

### Training

```bash
cd src
python train.py
```

### Testing

```bash
cd src
python test.py
```

### Interactive Analysis

Launch Jupyter notebook:

```bash
jupyter notebook src/notebooks/mnist-analysis.ipynb
```

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- O'Reilly: "Programming PyTorch for Deep Learning"

## Notes

- Model checkpoints saved in `models/`
- Training history saved as JSON
- Uses GPU acceleration when available
```
