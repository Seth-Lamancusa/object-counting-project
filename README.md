# CNN Educational Project

A minimal CNN (Convolutional Neural Network) project for educational purposes.

## Structure

- `model.py` - CNN architecture definition
- `data_loader.py` - Data loading utilities
- `train.py` - Training utilities and functions
- `main.py` - Main script to run training

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your data directory:
```
data/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            ...
    test/  (or val/)
        class1/
            img1.jpg
            ...
        class2/
            img1.jpg
            ...
```

3. Update `data_dir` in `main.py` if your data is in a different location

4. Run training:
```bash
python main.py
```

## Usage

The project loads images from a custom data directory. Make sure your data is organized in the structure shown above.

You can modify:
- Model architecture in `model.py`
- Data loading in `data_loader.py`
- Training parameters in `main.py`

